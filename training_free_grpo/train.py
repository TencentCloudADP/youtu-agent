import argparse
import asyncio
import copy
import json
import os
import random

from training_free_grpo.main import rollout_dataset, load_rollouts
from utu.agents import SimpleAgent
from utu.config import ConfigLoader

random.seed(42)


async def main(args):
    # Optional: override model via CLI before anything uses the LLM
    if args.model or args.base_url or args.api_key or args.local or args.provider or args.engine or args.weights:
        # Set env to let downstream LLM wrapper pick them up
        if args.local:
            # Set sensible defaults for local inference
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
        from training_free_grpo.math.verify import verify_func
        from training_free_grpo.math.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.math.experience import ExperienceUpdater
        config_name = "simple/math_agent.yaml"
    elif args.domain == "web":
        from training_free_grpo.web.dataset import load_data
        from training_free_grpo.web.verify import verify_func
        from training_free_grpo.web.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.web.experience import ExperienceUpdater
        config_name = "simple/base_search.yaml"
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")
    
    # Create experiment directory
    experiment_dir = os.path.join("data", args.domain, "train", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Set up the agent
    if args.mode == "prompt":
        worker_agent = None
    elif args.mode == "agent":
        config = ConfigLoader.load_agent_config(config_name)
        config.model.model_settings.temperature = args.rollout_temperature
        worker_agent = SimpleAgent(config=config)
        await worker_agent.build()
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")

    # Load the dataset (supports local file path for math datasets when ending with .json/.jsonl)
    train_data = load_data(args.dataset)
    print(f"Loaded {len(train_data)} records from dataset")
    if args.dataset_truncate is not None:
        print(f"- truncated to {args.dataset_truncate}")
        train_data = train_data[: args.dataset_truncate]
    assert len(train_data) % args.batchsize == 0

    # Set up the stats
    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        stats = json.load(open(stats_filename))
    else:
        stats = {}

    # Train
    for epoch in range(args.epochs):
        # Init
        print("=" * 30 + f"\nEpoch {epoch}\n" + "=" * 30)
        cur_epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(cur_epoch_dir, exist_ok=True)

        # Check if shuffled data already exists for this epoch
        shuffled_filename = os.path.join(cur_epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_filename):
            shuffled_data = []
            with open(shuffled_filename) as f:
                for line in f:
                    shuffled_data.append(json.loads(line))
            print(f"Loaded {len(shuffled_data)} records from shuffled data")
        else:
            print(f"Shuffling data ...")
            shuffled_data = copy.deepcopy(train_data)
            random.shuffle(shuffled_data)
            with open(shuffled_filename, "w") as f:
                for each in shuffled_data:
                    f.write(json.dumps(each) + "\n")

        # for each batch
        num_batches = len(shuffled_data) // args.batchsize
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            if f"step_{step}" not in stats:
                stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}
            elif stats[f"step_{step}"]["complete"]:
                continue

            # Init
            print(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)
            
            # Get current batch data
            batch_data = copy.deepcopy(shuffled_data[batch_idx * args.batchsize : (batch_idx + 1) * args.batchsize])

            # Load existing rollouts
            rollout_filename = os.path.join(cur_step_dir, "rollout.jsonl")
            rollouts = load_rollouts(rollout_filename)
            
            # Retrieve experiences for this batch (except first step)
            if step > 0:
                experience_filename = os.path.join("data", args.domain, "train", args.experiment_name, f"step_{step}/experiences.json")
                experiences = json.load(open(experience_filename))
            else:
                experiences = {}
            
            # Format the batch data with experiences
            formatted_experiences = "\n".join([ f"[{i}]. {e}" for i, e in experiences.items() ])
            formatted_batch_data = [{
                "prompt": PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                    experiences=formatted_experiences if formatted_experiences else "None",
                    problem=each["problem"],
                ) if experiences else each["problem"],
                **each
            } for each in batch_data]
            
            # Duplicate for GRPO
            print(f"GRPO rollout number={args.grpo_n}")
            formatted_batch_data = formatted_batch_data * args.grpo_n

            # Rollout the dataset
            rollouts, rollout_stats = await rollout_dataset(
                worker_agent=worker_agent,
                data=formatted_batch_data,
                rollouts=rollouts,
                verify_func=verify_func,
                rollout_filename=rollout_filename,
                rollout_concurrency=args.rollout_concurrency,
                task_timeout=args.task_timeout,
                temperature=args.rollout_temperature,
                max_tokens=args.rollout_max_tokens,
            )
            stats[f"step_{step}"]["rollout"] = rollout_stats

            # Generate critiques and update experiences
            next_step_dir = os.path.join(experiment_dir, f"step_{step+1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_experience_filename = os.path.join(next_step_dir, "experiences.json")
            if os.path.exists(next_experience_filename):
                print(f"Experiences already exist for step {step}, skipping experience update")
            else:
                new_experiences = ExperienceUpdater().run(
                    rollouts=rollouts, 
                    experiences=experiences,
                    save_dir=cur_step_dir,
                    max_workers=args.rollout_concurrency,
                    given_ground_truth=True if args.given_ground_truth=="True" else False,
                    only_partial_correct=True if args.grpo_n > 1 else False,
                )
                json.dump(new_experiences, open(next_experience_filename, "w"), indent=2)
                print(f"Saved {len(new_experiences)} experiences to {next_experience_filename}")

            # Save stats
            stats[f"step_{step}"]["complete"] = True
            json.dump(stats, open(stats_filename, "w"), indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-free GRPO")
    parser.add_argument("--mode", type=str, default="agent", required=True, choices=["prompt", "agent"], help="Mode of inference")
    parser.add_argument("--domain", type=str, required=True, choices=["math", "web"], help="domain of the tasks (math/web)")
    parser.add_argument("--experiment_name", type=str, required=True, help="name of experiment run")
    parser.add_argument("--dataset", type=str, required="True", help="Name of dataset")
    parser.add_argument("--dataset_truncate", type=int, default=None, help="Truncate dataset to first N samples")
    parser.add_argument("--given_ground_truth", type=str, default="True", help="Whether use ground truth answers")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument("--batchsize", type=int, default=64, help="batchsize")
    parser.add_argument("--grpo_n", type=int, default=5, help="number of rollouts in a group of GRPO")
    parser.add_argument("--rollout_concurrency", type=int, default=5, help="Concurrency level for rollouts")
    parser.add_argument("--rollout_temperature", type=float, default=0.7, help="Temperature for the LLM")
    parser.add_argument("--rollout_max_tokens", type=int, default=16384, help="Max tokens for each rollout batch")
    parser.add_argument("--task_timeout", type=float, default=3600, help="Timeout for each individual task in seconds")
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
