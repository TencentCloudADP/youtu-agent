"""
Experience updater for training-free GRPO.
"""

import asyncio
import copy
import json
import re
from collections import defaultdict

from agents import custom_span
from tqdm import tqdm

from ..config import AgentConfig
from ..db import EvaluationSample
from ..utils import FileUtils, SimplifiedAsyncOpenAI, get_logger
from .utils import TaskRecorder

logger = get_logger(__name__)


class ExperienceUpdater:
    def __init__(self, config: AgentConfig, agent_objective: str, learning_objective: str):
        self.config = config
        self.agent_objective = agent_objective
        self.learning_objective = learning_objective
        self.prompts = FileUtils.load_prompts("practice/experience.yaml")
        self.llm = SimplifiedAsyncOpenAI(**config.model.model_provider.model_dump())

    async def run(
        self,
        rollouts: list[EvaluationSample],
        recorder: TaskRecorder,
        concurrency: int = 16,
        given_ground_truth: bool = True,
        num_experiences: int = 2,
    ) -> None:
        """Update experiences based on rollouts."""
        # 1. Summarize trajectory for each rollout
        with custom_span("Trajectory Summarization"):
            problem_to_summarized_rollouts = await self._single_rollout_summary(
                rollouts=rollouts, concurrency=concurrency, given_ground_truth=given_ground_truth
            )

        # 2. Generate semantic group advantages based on summarized rollouts
        with custom_span("Semantic Group Advantage"):
            new_experiences = await self._group_advantage(
                problem_to_summarized_rollouts=problem_to_summarized_rollouts,
                concurrency=concurrency,
                given_ground_truth=given_ground_truth,
                num_experiences=num_experiences,
            )

        # 3. group update experiences
        with custom_span("Group update"):
            critiques = await self._group_update(
                recorder=recorder,
                new_experiences=new_experiences,
                concurrency=concurrency,
            )

        # 4. batch update experiences
        with custom_span("Batch update"):
            new_experiences = await self._batch_update(
                recorder=recorder,
                critiques=critiques,
            )

        # 5. assign new experience IDs
        new_experiences = {f"G{i}": exp for i, exp in enumerate(new_experiences.values())}
        recorder.experiences_update(new_experiences)
        return new_experiences

    async def _single_rollout_summary(
        self,
        rollouts: list[EvaluationSample],
        concurrency: int,
        given_ground_truth: bool,
    ) -> dict[str, list[str]]:
        """Summarize each rollout's trajectory."""
        # group by problems
        problems_to_rollouts = defaultdict(list)
        for rollout in rollouts:
            if len(rollout.trajectories) > 0:
                problems_to_rollouts[rollout.raw_question].append(rollout)

        # only summarize the group whose rollouts are partially correct
        all_rollouts_to_process = []
        for rollouts in problems_to_rollouts.values():
            if given_ground_truth:
                # only for those partially correct
                scores = [each.reward for each in rollouts]
                avg_score = sum(scores) / len(scores)
                if avg_score > 0 and avg_score < 1:
                    all_rollouts_to_process.extend(rollouts)
            else:
                all_rollouts_to_process.extend(rollouts)

        semaphore = asyncio.Semaphore(concurrency)

        async def summarize_with_semaphore(item: EvaluationSample):
            async with semaphore:
                try:
                    with custom_span("summary single rollout"):
                        sp = FileUtils.get_jinja_template_str(
                            self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP"]
                        ).render(
                            agent_objective=self.agent_objective,
                            learning_objective=self.learning_objective,
                        )
                        up = FileUtils.get_jinja_template_str(
                            self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP"]
                        ).render(
                            question=item.raw_question,
                            trajectory=json.loads(item.trajectories)[0]["trajectory"],
                            answer=item.correct_answer if given_ground_truth else "[REDACTED]",
                            critique=item.reasoning or "[No critique provided]",
                        )
                        response = await self.llm.query_one(
                            messages=[
                                {"role": "system", "content": sp},
                                {"role": "user", "content": up},
                            ],
                            **self.config.model.model_params.model_dump(),
                        )
                    return {"trajectory_summary": response, **item.model_dump()}
                except Exception as e:
                    logger.warning(f"Warning: failed in single rollout summary, {e}")
                    return None

        # parallel running
        tasks = [summarize_with_semaphore(item) for item in all_rollouts_to_process]
        results = defaultdict(list)
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Single rollout summary"):
            result = await task
            if result is not None:
                problem = result["raw_question"]
                results[problem].append(result)
        return results

    async def _group_advantage(
        self,
        problem_to_summarized_rollouts: dict[str, list[dict]],
        concurrency: int,
        given_ground_truth: bool,
        num_experiences: int,
    ) -> dict[str, dict]:
        """Generate critique for each query based on summarized rollouts."""
        all_rollouts = []
        for rollouts in problem_to_summarized_rollouts.values():
            if given_ground_truth:
                # only for those partially correct
                scores = [each["reward"] for each in rollouts]
                avg_score = sum(scores) / len(scores)
                if avg_score > 0 and avg_score < 1:
                    all_rollouts.append(rollouts)
            else:
                all_rollouts.append(rollouts)

        semaphore = asyncio.Semaphore(concurrency)

        async def critique_with_semaphore(rollouts_per_problem: list[dict]):
            async with semaphore:
                try:
                    with custom_span("single query group advantage"):
                        formatted_trajectories = "\n\n".join(
                            [
                                f"Attempt {i + 1} (Reward {each['reward'] if given_ground_truth else '[REDACTED]'}):\n"
                                f"{each['trajectory_summary']}"
                                for i, each in enumerate(rollouts_per_problem)
                            ]
                        )
                        sp = FileUtils.get_jinja_template_str(self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_SP"]).render(
                            agent_objective=self.agent_objective,
                            learning_objective=self.learning_objective,
                            num_experiences=num_experiences,
                        )
                        up = FileUtils.get_jinja_template_str(self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_UP"]).render(
                            question=rollouts_per_problem[0]["raw_question"],
                            answer=rollouts_per_problem[0]["correct_answer"] if given_ground_truth else "[REDACTED]",
                            trajectories=formatted_trajectories,
                        )
                        response = await self.llm.query_one(
                            messages=[
                                {"role": "system", "content": sp},
                                {"role": "user", "content": up},
                            ],
                            **self.config.model.model_params.model_dump(),
                        )

                        # extract experiences from the response
                        pattern = re.compile(r"<Experiences>\s*(.*?)\s*</Experiences>", re.DOTALL | re.IGNORECASE)
                        match = pattern.search(response)
                        experiences = match.group(1).strip() if match else ""
                    return {"rollouts": rollouts_per_problem, "critique": response, "experiences": experiences}
                except Exception as e:
                    logger.warning(f"Warning: failed in single group advantage, {e}")
                    return None

        # parallel running
        results = []
        tasks = [critique_with_semaphore(rollouts_per_problem) for rollouts_per_problem in all_rollouts]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Single query group advantage"):
            result = await task
            if result is not None:
                results.append(result)

        return results

    async def _group_update(
        self,
        recorder: TaskRecorder,
        new_experiences: list[dict],
        concurrency: int,
    ) -> dict[str, str]:
        """Group update experiences based on critiques."""
        semaphore = asyncio.Semaphore(concurrency)

        async def group_update_with_semaphore(new_experience: dict):
            async with semaphore:
                try:
                    with custom_span("single group update"):
                        # get current experiences from recorder
                        curr_experiences = recorder.experiences or {}
                        formatted_experiences = (
                            "\n".join([f"[{i}]. {e}" for i, e in curr_experiences.items()])
                            if curr_experiences
                            else "None"
                        )
                        sp = FileUtils.get_jinja_template_str(
                            self.prompts["GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP"]
                        ).render(
                            agent_objective=self.agent_objective,
                            learning_objective=self.learning_objective,
                        )
                        up = FileUtils.get_jinja_template_str(
                            self.prompts["GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP"]
                        ).render(
                            existing_experiences=formatted_experiences,
                            new_experiences=new_experience["experiences"],
                        )
                        response = await self.llm.query_one(
                            messages=[
                                {"role": "system", "content": sp},
                                {"role": "user", "content": up},
                            ],
                            **self.config.model.model_params.model_dump(),
                        )
                        # parse response
                        response = response.split("```json")[-1].split("```")[0]
                        operations = json.loads(response)
                    return {"operations": operations, **new_experience}
                except Exception as e:
                    logger.warning(f"Warning: failed in group update experience, {e}")
                    return None

        # parallel running
        results = []
        tasks = [group_update_with_semaphore(new_experience) for new_experience in new_experiences]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Group update"):
            result = await task
            if result is not None:
                results.append(result)
        return results

    async def _batch_update(
        self, recorder: TaskRecorder, critiques: list[dict], max_retries: int = 3
    ) -> dict[str, dict]:
        """Batch update experiences based on critiques."""
        # get current experiences from recorder
        logger.info("Batch update")
        # collect operations
        all_operations = []
        for each in critiques:
            all_operations.extend(each["operations"])
        print("- Num of operations to process:", len(all_operations))

        # use LLM to get the revision plan
        experiences = recorder.experiences or {}
        revision_plan = []
        for _ in range(max_retries):
            try:
                sp = FileUtils.get_jinja_template_str(self.prompts["BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP"]).render(
                    agent_objective=self.agent_objective,
                    learning_objective=self.learning_objective,
                )
                up = FileUtils.get_jinja_template_str(self.prompts["BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP"]).render(
                    experiences_and_operations=self._format_exp_and_ops(experiences, all_operations)
                )
                response = await self.llm.query_one(
                    messages=[
                        {"role": "system", "content": sp},
                        {"role": "user", "content": up},
                    ],
                    **self.config.model.model_params.model_dump(),
                )
                # parse response
                revision_plan = json.loads(response.split("```json")[-1].split("```")[0])
                break
            except Exception:
                print("Warning: failed to decode in updating general experiences")

        # apply revision plan to get new experiences
        max_ID = len(experiences)
        new_experiences = copy.deepcopy(experiences)
        for plan in revision_plan:
            operation = plan.get("operation", "ADD")
            content = plan.get("content", "")
            target_id = plan.get("id", None)
            if not content:
                continue

            if operation == "ADD":
                new_experiences[f"{max_ID}"] = content
                max_ID += 1
            elif operation == "UPDATE":
                if target_id in new_experiences:
                    new_experiences[target_id] = content
                else:
                    # directly add new experience
                    new_experiences[f"{max_ID}"] = content
                    max_ID += 1
            elif operation == "DELETE":
                if target_id in new_experiences:
                    del new_experiences[target_id]
        print("- Num of candidate experiences:", len(new_experiences))
        return new_experiences

    def _format_exp_and_ops(self, experiences: dict[str, str], operations: list[dict]) -> str:
        """Format experiences and operations."""
        if not operations:
            return "No batch operations."

        # Format existing experiences and their related operations
        formatted_res = []
        for id, exp in experiences.items():
            curr_str = f"Experience {id}:\nContent: {exp}\n"
            related_ops = [op for op in operations if op.get("id") == id]
            if related_ops:
                curr_str += "Related Operations:\n"
                op_str = []
                for op in related_ops:
                    op_str.append(f"{json.dumps(op, ensure_ascii=False, indent=2)}")
                op_str = "\n".join(op_str)
                curr_str += op_str
            else:
                curr_str += "No related operations."
            formatted_res.append(curr_str)

        # Format operations without specific IDs
        no_id_ops = [op for op in operations if not op.get("id", None)]
        if no_id_ops:
            curr_str = "Operations without specific Experience ID:\n"
            op_str = []
            for op in no_id_ops:
                op_str.append(f"{json.dumps(op, ensure_ascii=False, indent=2)}")
            op_str = "\n".join(op_str)
            curr_str += op_str
            formatted_res.append(curr_str)

        return "\n\n".join(formatted_res)
