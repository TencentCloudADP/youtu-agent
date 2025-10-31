"""
CLI usage: python scripts/cli_chat.py --config examples/paper_collector
"""

import asyncio
import pathlib

from utu.agents import OrchestratorAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils, FileUtils


async def main():
    config = ConfigLoader.load_agent_config("examples/paper_collector")
    runner = OrchestratorAgent(config)

    data_dir = pathlib.Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    paper_url = "https://www.arxiv.org/pdf/2507.12883"
    question = f"请分析论文{paper_url}，整理出它的相关工作，并且进行简单的比较。"

    result = runner.run_streamed(question)
    await AgentsUtils.print_stream_events(result.stream_events())

    print(f"Run completed with result: {result}")
    FileUtils.save_json(data_dir / "trajectories.json", result.trajectories)
    with open(data_dir / "final_output.txt", "w", encoding="utf-8") as f:
        f.write(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
