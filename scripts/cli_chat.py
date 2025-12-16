import asyncio

import art

from utu.agents import OrchestratorAgent, SimpleAgent, get_agent
from utu.utils import AgentsUtils, PrintUtils
from utu.utils.script_utils import parse_cli_args

USAGE_MSG = f"""{"-" * 100}
Usage: `python cli_chat.py --config_name <config_name>`
Quit: exit, quit, q
{"-" * 100}""".strip()


async def main():
    text = art.text2art("Youtu-agent", font="small")
    PrintUtils.print_info(text, color="blue")
    PrintUtils.print_info(USAGE_MSG, color="yellow")

    config = parse_cli_args()
    agent = get_agent(config=config)
    if isinstance(agent, SimpleAgent):
        await agent.build()
    history = None
    while True:
        user_input = await PrintUtils.async_print_input("> ")
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            break
        if not user_input.strip():
            continue

        if isinstance(agent, SimpleAgent):
            await agent.chat_streamed(user_input)
        elif isinstance(agent, OrchestratorAgent):
            history = agent.run_streamed(user_input, history)
            await AgentsUtils.print_stream_events(history.stream_events())
        else:
            PrintUtils.print_error(f"Unsupported agent type: {type(agent)}")
            return
    if isinstance(agent, SimpleAgent):
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
