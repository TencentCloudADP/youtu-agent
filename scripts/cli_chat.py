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
    parser.add_argument("--config_name", type=str, default="default", help="Configuration name")
    parser.add_argument("--agent_model", type=str, default=None, help="Agent model.")
    parser.add_argument("--tools", type=str, nargs="*", help="List of tool names to load")
    parser.add_argument("--stream", action="store_true", help="Stream the output.")
    args = parser.parse_args()

    config: AgentConfig = ConfigLoader.load_agent_config(args.config_name)
    if args.agent_model:
        config.model.model = args.agent_model
    if args.tools:
        # load toolkits from config
        config.toolkits = []
        for tool_name in args.tools:
            config.toolkits.append(ConfigLoader.load_toolkit_config(tool_name))

    async with SimpleAgent(config=config) as agent:
        while True:
            user_input = input("> ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if args.stream:
                await agent.chat_streamed(user_input)
            else:
                await agent.chat(user_input)


if __name__ == "__main__":
    asyncio.run(main())
