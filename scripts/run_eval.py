import argparse
import asyncio

from utu.config import ConfigLoader, EvalConfig
from utu.eval import BaseBenchmark


def get_eval_config(args: argparse.Namespace) -> EvalConfig:
    config = ConfigLoader.load_eval_config(args.config_name)
    if args.exp_id:
        config.exp_id = args.exp_id
    if args.agent_model:
        config.agent.model.model_provider.model = args.agent_model
    if args.dataset:
        config.data.dataset = args.dataset
    if args.dataset_type:
        config.data.type = args.dataset_type
    if args.concurrency:
        config.concurrency = args.concurrency
    if args.judge_concurrency:
        config.judge_concurrency = args.judge_concurrency
    return config


async def main():
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument("--config_name", type=str, default="ww", help="Configuration name for evaluation.")
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment ID.")
    parser.add_argument("--agent_model", type=str, default=None, help="Agent model.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset.")
    parser.add_argument("--dataset_type", type=str, default=None, help="Dataset type.")
    parser.add_argument("--concurrency", type=int, default=None, help="Test concurrency.")
    parser.add_argument("--judge_concurrency", type=int, default=None, help="Judge concurrency.")

    # eval steps
    parser.add_argument(
        "--step", type=str, default="all", choices=["all", "rollout", "judge"], help="Evaluation step to run."
    )
    args = parser.parse_args()

    config = get_eval_config(args)

    runner = BaseBenchmark(config)
    match args.step:
        case "all":
            await runner.main()
        case "rollout":
            runner.preprocess()
            await runner.rollout()
        case "judge":
            await runner.judge(stage="rollout")  # set stage=None to rejudge; rollout or judged incrementally
            await runner.stat()
        case _:
            raise ValueError(f"Unsupported stage: {args.step}")


if __name__ == "__main__":
    asyncio.run(main())
