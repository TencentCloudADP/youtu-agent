"""
- [x] setup tracing
- [x] purify logging
- [ ] support stream?
"""

from agents import trace

from ..config import AgentConfig, ConfigLoader
from ..utils import AgentsUtils, get_logger
from .workforce import AnswererAgent, AssignerAgent, ExecutorAgent, PlannerAgent, WorkspaceTaskRecorder

logger = get_logger(__name__)


class WorkforceAgent:
    name = "workforce_agent"

    def __init__(self, config: AgentConfig | str):
        """Initialize the workforce agent"""
        if isinstance(config, str):
            config = ConfigLoader.load_agent_config(config)
        self.config = config

    async def run(self, input: str, trace_id: str = None) -> WorkspaceTaskRecorder:
        trace_id = trace_id or AgentsUtils.gen_trace_id()

        logger.info("Initializing agents...")
        planner_agent = PlannerAgent(config=self.config)
        assigner_agent = AssignerAgent(config=self.config)
        answerer_agent = AnswererAgent(config=self.config)
        executor_agent_group: dict[str, ExecutorAgent] = {}
        for name, config in self.config.workforce_executor_agents.items():
            executor_agent_group[name] = ExecutorAgent(config=config, workforce_config=self.config)

        recorder = WorkspaceTaskRecorder(
            overall_task=input, executor_agent_kwargs_list=self.config.workforce_executor_infos
        )

        with trace(workflow_name=self.name, trace_id=trace_id):
            # * 1. generate plan
            logger.info("Generating plan...")
            await planner_agent.plan_task(recorder)
            logger.info(f"Plan: {recorder.task_plan}")

            # DISCUSS: merge .get_next_task and .has_uncompleted_tasks? (while True)
            while recorder.has_uncompleted_tasks:
                # * 2. assign tasks
                next_task = await assigner_agent.assign_task(recorder)
                logger.info(f"Assign task: {next_task.task_id} assigned to {next_task.assigned_agent}")

                # * 3. execute task
                logger.info(f"Executing task: {next_task.task_id}")
                await executor_agent_group[next_task.assigned_agent].execute_task(recorder=recorder, task=next_task)
                logger.info(f"Task {next_task.task_id} result: {next_task.task_result}")
                await planner_agent.plan_check(recorder, next_task)
                logger.info(f"Task {next_task.task_id} checked: {next_task.task_status}")

                # * 4. update plan
                if not recorder.has_uncompleted_tasks:  # early stop
                    break
                plan_update_choice = await planner_agent.plan_update(recorder, next_task)
                logger.info(f"Plan update choice: {plan_update_choice}")
                if plan_update_choice == "stop":
                    logger.info("Planner determined overall task is complete, stopping execution")
                    break
                elif plan_update_choice == "update":
                    logger.info(f"Task plan updated: {recorder.task_plan}")

            final_answer = await answerer_agent.extract_final_answer(recorder)
            logger.info(f"Extracted final answer: {final_answer}")
            recorder.set_final_output(final_answer)

            # TODO: self-eval, which will be used in next attempt!
            # success = await answerer_agent.answer_check(
            #     question=agent_workspace.overall_task,
            #     model_answer=final_answer,
            #     ground_truth=task["Final answer"]
            # )
        return recorder
