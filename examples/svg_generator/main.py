"""
CLI usage: python scripts/cli_chat.py --config examples/svg_generator
"""

import asyncio

from utu.agents import OrchestratorAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils


async def main():
    config = ConfigLoader.load_agent_config("examples/svg_generator")
    runner = OrchestratorAgent(config)

    question = "deepseek-v3.1有哪些亮点更新?"

    res = runner.run_streamed(question)
    await AgentsUtils.print_stream_events(res.stream_events())
    print(f"Final output: {res.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
