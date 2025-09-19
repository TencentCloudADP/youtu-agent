import argparse
import asyncio

import art

from utu.agents import SimpleAgent
from utu.config import AgentConfig, ConfigLoader
from utu.utils import PrintUtils

USAGE_MSG = f"""{"-" * 100}
Usage: python cli_chat.py --config_name <config_name>
Quit: exit, quit, q
{"-" * 100}""".strip()


async def main():
    text = art.text2art("Youtu-agent", font="small")
    PrintUtils.print_info(text, color="blue")
    PrintUtils.print_info(USAGE_MSG, color="yellow")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="simple/base", help="Configuration name")
    parser.add_argument("--agent_model", type=str, default=None, help="Agent model.")
    parser.add_argument("--tools", type=str, nargs="*", help="List of tool names to load")
    args = parser.parse_args()

    config: AgentConfig = ConfigLoader.load_agent_config(args.config_name)

    agent = SimpleAgent(config=config)
    while True:
        user_input = await PrintUtils.async_print_input("> ")
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            break
        if not user_input.strip():
            continue
        if hasattr(agent, "build"):
            await agent.build()
        # TODO: use a unified .run_streamed interface
        await agent.chat_streamed(user_input)


if __name__ == "__main__":
    asyncio.run(main())
