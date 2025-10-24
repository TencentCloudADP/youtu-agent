"""
CLI usage: python scripts/cli_chat.py --config examples/data_analysis
"""

import asyncio
import pathlib
import re

from utu.agents import OrchestratorAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils, FileUtils


async def main():
    config = ConfigLoader.load_agent_config("examples/data_analysis")
    runner = OrchestratorAgent(config)

    # Run the agent with a sample question
    # data from https://www.kaggle.com/datasets/joannanplkrk/its-raining-cats
    fn = pathlib.Path(__file__).parent / "demo_data_cat_breeds_clean.csv"
    assert fn.exists(), f"File {fn} does not exist."
    question = f"请分析位于`{fn}`的猫品种数据，提取有价值的信息。"

    result = runner.run_streamed(question)
    await AgentsUtils.print_stream_events(result.stream_events())

    # Extract and print the result
    FileUtils.save_json(fn.parent / "trajectories.json", result.trajectories)
    with open(fn.parent / "report.html", "w", encoding="utf-8") as f:
        match = re.search(r"```html(.*?)```", result.final_output, re.DOTALL)
        html_str = match.group(1).strip() if match else result.final_output
        f.write(html_str)


if __name__ == "__main__":
    asyncio.run(main())
