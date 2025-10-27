import os
import json
import argparse
import asyncio
import copy
import time
import traceback

from tqdm import tqdm
from collections import defaultdict

from utu.agents import SimpleAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils
from utu.agents.common import TaskRecorder
from training_free_grpo.llm import LLM


def load_rollouts(rollout_filename: str) -> list[dict]:
    results = []
    if os.path.exists(rollout_filename):
        with open(rollout_filename, encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
    return results


def save_rollouts(results: list[dict], rollout_filename: str):
    with open(rollout_filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


async def rollout_dataset(
    worker_agent: SimpleAgent | None,
    data: list[dict],
    rollouts: list[dict],
    rollout_filename: str,
    verify_func: callable,
    rollout_concurrency: int = 5,
    task_timeout: float = 3600,
    max_retries: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 16384,
) -> list[dict]:
    """Rollout the dataset using the worker agent with concurrency control, timeout, error handling, and retries."""

    # examine data and existing rollouts
    if len(rollouts) > 0:
        for each in rollouts:
            assert "runid" in each
        data_problems = [each["problem"] for each in data]
        rollouts_problems = [each["problem"] for each in rollouts]
        assert data_problems == rollouts_problems, (
            f"The problems in data should be the same as existing rollouts {rollout_filename}"
        )
    else:
        for sample in data:
            assert "problem" in sample and "groundtruth" in sample
        rollouts = [{"runid": i, **sample} for i, sample in enumerate(data)]
    save_rollouts(rollouts, rollout_filename)

    # create task queue
    task_queue = asyncio.Queue()
    pending_tasks_count = 0
    for sample in rollouts:
        if "trajectories" not in sample or len(sample["trajectories"]) == 0:
            sample_with_retry = copy.deepcopy(sample)
            sample_with_retry["retry_count"] = 0
            await task_queue.put(sample_with_retry)
            pending_tasks_count += 1
    pbar = tqdm(total=pending_tasks_count, desc="Rolling out")

    async def worker(name: str):
        while not task_queue.empty():
            sample = await task_queue.get()
            task_start_time = time.time()
            try:
                if worker_agent is None:
                    llm = LLM()
                    coro = asyncio.to_thread(llm.chat, sample["prompt"], temperature=temperature, max_tokens=max_tokens)
                    res = await asyncio.wait_for(coro, timeout=task_timeout)                    
                    res = TaskRecorder(
                            final_output=res,
                            trajectories=[{
                                "trajectory": [
                                    {"role": "user", "content": sample["prompt"]},
                                    {"role": "assistant", "content": res}
                                ]
                            }],
                        )
                else:
                    async with worker_agent as agent:
                        async def rollout_streamed(sample) -> TaskRecorder:
                            prompt = sample.get("prompt", sample["problem"])
                            res = agent.run_streamed(prompt)
                            async for _ in res.stream_events(): pass
                            traj = AgentsUtils.get_trajectory_from_agent_result(res)
                            return TaskRecorder(
                                final_output=res.final_output,
                                trajectories=[traj],
                            )
                        res = await asyncio.wait_for(rollout_streamed(sample), timeout=task_timeout)
                
                task_end_time = time.time()
                sample.update(
                    {
                        "response": res.final_output,
                        "trajectories": res.trajectories,
                        "error": None,
                        "rollout_time": task_end_time - task_start_time,
                    }
                )
                sample["reward"] = verify_func(sample, sample["groundtruth"])
                
                # Task succeeded
                rollouts[sample["runid"]] = sample
                save_rollouts(rollouts, rollout_filename)
                pbar.update(1)

            except Exception as e:
                task_end_time = time.time()
                sample["retry_count"] += 1
                error_info = traceback.format_exc()
                print(f"> error: {error_info}")
                
                if sample["retry_count"] <= max_retries:
                    tqdm.write(f"Worker {name}: Task runid={sample['runid']} failed with {type(e).__name__}. Retrying ({sample['retry_count']}/{max_retries})...")
                    await task_queue.put(sample) # Re-queue the task
                else:
                    tqdm.write(f"Worker {name}: Task runid={sample['runid']} failed after {max_retries} retries. Error: {e}. Traceback: {error_info}")
                    sample.update(
                        {
                            "response": f"Error: {str(e)} after {max_retries} retries.",
                            "trajectories": [],
                            "error": error_info,
                            "reward": 0,
                            "rollout_time": task_end_time - task_start_time,
                        }
                    )
                    
                    # Task failed permanently
                    rollouts[sample["runid"]] = sample
                    save_rollouts(rollouts, rollout_filename)
                    pbar.update(1)
            finally:
                task_queue.task_done()

    # run all tasks
    workers = [asyncio.create_task(worker(f"worker-{i}")) for i in range(rollout_concurrency)]
    await task_queue.join()

    # clean up
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    pbar.close()
    print(f"Successfully processed {len(rollouts)} samples.")

    # stats
    all_rewards = []
    problem_to_scores = defaultdict(list)
    num_tool_calls = []
    for rollout in rollouts:
        all_rewards.append(rollout.get("reward", 0))
        problem_to_scores[rollout["problem"]].append(rollout.get("reward", 0))
        if "trajectories" in rollout and rollout["trajectories"]:
            num_tool_calls.append(
                len([each for each in rollout["trajectories"][0]["trajectory"] if each["role"] == "tool"])
            )
    problem_to_max_score = {problem: max(scores) for problem, scores in problem_to_scores.items()}
    max_K = max((len(scores) for scores in problem_to_scores.values()), default=0)
    stats = {
        "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
        f"Pass@{max_K}": sum(max_reward > 0 for max_reward in problem_to_max_score.values()) / len(problem_to_max_score)
        if problem_to_max_score else 0,
        "avg_tool_call": sum(num_tool_calls) / len(num_tool_calls) if num_tool_calls else 0,
    }
    for k, v in stats.items():
        print(f"- {k}: {v}")
    return rollouts, stats


async def main(args):
    # Optional: override model via CLI before anything uses the LLM
    if args.model or args.base_url or args.api_key or args.local or args.provider or args.engine or args.weights:
        if args.local:
            os.environ.setdefault("UTU_LLM_BASE_URL", "http://localhost:8000/v1")
            os.environ.setdefault("UTU_LLM_API_KEY", "dummy")
        if args.provider:
            provider = args.provider.lower()
            if provider == "ollama":
                os.environ["UTU_LLM_BASE_URL"] = os.environ.get("UTU_LLM_BASE_URL", "http://localhost:11434/v1")
                os.environ.setdefault("UTU_LLM_API_KEY", "ollama")
            elif provider == "vllm":
                os.environ["UTU_LLM_BASE_URL"] = os.environ.get("UTU_LLM_BASE_URL", "http://localhost:8000/v1")
                os.environ.setdefault("UTU_LLM_API_KEY", "dummy")
        if args.model:
            os.environ["UTU_LLM_MODEL"] = args.model
        if args.base_url:
            os.environ["UTU_LLM_BASE_URL"] = args.base_url
        if args.api_key:
            os.environ["UTU_LLM_API_KEY"] = args.api_key
        if args.engine:
            os.environ["UTU_LLM_ENGINE"] = args.engine
        if args.weights:
            os.environ["UTU_LLM_WEIGHTS"] = args.weights
    # Set up domain-specific variables
    if args.domain == "math":
        from training_free_grpo.math.dataset import load_data
        if args.math_verifier == "llm":
            from training_free_grpo.math.verify_llm import verify_func
        else:
            from training_free_grpo.math.verify import verify_func
        from training_free_grpo.math.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        config_name = "simple/math_agent.yaml"
    elif args.domain == "web":
        from training_free_grpo.web.dataset import load_data
        from training_free_grpo.web.verify import verify_func
        from training_free_grpo.web.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        config_name = "simple/search_agent.yaml"
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")

    # Set up the agent
    if args.mode == "prompt":
        worker_agent = None
    elif args.mode == "agent":
        config = ConfigLoader.load_agent_config(config_name)
        # Apply sampling temperature if provided
        try:
            config.model.model_settings.temperature = args.rollout_temperature
            if args.top_p is not None:
                config.model.model_settings.top_p = args.top_p
            if args.top_k is not None:
                config.model.model_settings.top_k = args.top_k
        except Exception:
            pass
        worker_agent = SimpleAgent(config=config)
        await worker_agent.build()
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")

    # Load the dataset
    test_data = load_data(args.dataset)
    print(f"Loaded {len(test_data)} records from dataset")
    if args.dataset_truncate is not None:
        print(f"- truncated to {args.dataset_truncate}")
        test_data = test_data[: args.dataset_truncate]
    
    # Insert experiences
    if args.experience_file:
        experiences = json.load(open(args.experience_file))
        formatted_experiences = "\n".join([ f"[{i}]. {e}" for i, e in experiences.items() ])
        formatted_test_data = [{
            "prompt": PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                experiences=formatted_experiences if formatted_experiences else "None",
                problem=each["problem"],
            ),
            **each
        } for each in test_data]
    else:
        formatted_test_data = [{
            "prompt": each["problem"],
            **each
        } for each in test_data]
    
    # Duplicate for Pass@k evaluation
    formatted_test_data = formatted_test_data * args.pass_k
    print(f"Duplicated to {len(formatted_test_data)} records for Pass@{args.pass_k} evaluation")

    # Load existing rollouts
    os.makedirs(f"data/{args.domain}/eval", exist_ok=True)
    rollout_filename = f"data/{args.domain}/eval/{args.experiment_name}.jsonl"
    rollouts = load_rollouts(rollout_filename)

    # Rollout the dataset
    # Wire top_p and top_k via temperature path using the LLM wrapper (HTTP or vLLM)
    await rollout_dataset(
        worker_agent=worker_agent,
        data=formatted_test_data,
        rollouts=rollouts,
        verify_func=verify_func,
        rollout_filename=rollout_filename,
        rollout_concurrency=args.rollout_concurrency,
        task_timeout=args.task_timeout,
        temperature=args.rollout_temperature,
        max_tokens=args.rollout_max_tokens,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-Free GRPO Evaluation")
    parser.add_argument("--mode", type=str, default="agent", required=True, choices=["prompt", "agent"], help="Mode of inference")
    parser.add_argument("--domain", type=str, required=True, choices=["math", "web"], help="The domain of the experiment")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment run")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset")
    parser.add_argument("--dataset_truncate", type=int, default=None, help="Truncate dataset to first N samples")
    parser.add_argument("--experience_file", type=str, default=None)
    parser.add_argument("--rollout_concurrency", type=int, default=5, help="Concurrency level for rollouts")
    parser.add_argument("--rollout_temperature", type=float, default=0.3, help="Temperature for sampling")
    parser.add_argument("--rollout_max_tokens", type=int, default=16384, help="Max tokens for each rollout")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling (optional)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (optional)")
    parser.add_argument("--pass_k", type=int, default=1, help="Pass@k metric")
    parser.add_argument("--task_timeout", type=float, default=3600, help="Timeout for each individual task in seconds")
    parser.add_argument("--math_verifier", type=str, default="rule", choices=["rule", "llm"], help="Math verification backend")
    # Optional model overrides (for local model endpoints)
    parser.add_argument("--model", type=str, default=None, help="Override: LLM model name (env UTU_LLM_MODEL)")
    parser.add_argument("--base_url", type=str, default=None, help="Override: LLM base URL (env UTU_LLM_BASE_URL)")
    parser.add_argument("--api_key", type=str, default=None, help="Override: LLM API key (env UTU_LLM_API_KEY)")
    parser.add_argument("--local", action="store_true", help="Use local model defaults (base_url=http://localhost:8000/v1, api_key=dummy)")
    parser.add_argument("--provider", type=str, default=None, choices=["ollama", "vllm"], help="Shortcut for local providers")
    # Native engines
    parser.add_argument("--engine", type=str, default=None, choices=["openai", "vllm"], help="Engine backend (default: openai)")
    parser.add_argument("--weights", type=str, default=None, help="Local weights folder for engine=vllm (env UTU_LLM_WEIGHTS)")

    args = parser.parse_args()
    asyncio.run(main(args))