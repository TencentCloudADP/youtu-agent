"""
CLI usage: python scripts/cli_chat.py --config examples/deep_learning_trainer
"""

import asyncio

from utu.agents import OrchestratorAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils


async def main():
    config = ConfigLoader.load_agent_config("examples/deep_learning/deep_learning_trainer")
    runner = OrchestratorAgent(config)

    question = "基于configs/agents/examples/deep_learning目录下的训练代码，对mnist分类模型进行迭代优化。"

    res = runner.run_streamed(question)
    await AgentsUtils.print_stream_events(res.stream_events())
    print(f"Final output: {res.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
