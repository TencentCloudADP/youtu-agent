import argparse
import asyncio

from utu.agents import SimpleAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils, PrintUtils


async def run_agent(config_name: str, prompt: str) -> None:
    """Run a SimpleAgent once with streaming output."""
    config = ConfigLoader.load_agent_config(config_name)

    async with SimpleAgent(config=config) as agent:
        PrintUtils.print_info(f"Loaded agent `{config.agent.name}` – sending prompt...\n", color="green")
        recorder = agent.run_streamed(prompt, save=True)
        await AgentsUtils.print_stream_events(recorder.stream_events())
        PrintUtils.print_info("\n=== Final Answer ===", color="yellow")
        PrintUtils.print_bot(recorder.final_output or "[empty]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Youtu-Agent demo without CLI wrapper.")
    parser.add_argument(
        "--config_name",
        type=str,
        default="agents/examples/file_manager",
        help="Hydra-style config path (e.g. agents/examples/file_manager).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请使用 run_bash 执行 `ls -a` 并总结我桌面文件。",
        help="Initial user query for the agent.",
    )
    return parser.parse_args()


async def async_main():
    args = parse_args()
    await run_agent(config_name=args.config_name, prompt=args.prompt)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
