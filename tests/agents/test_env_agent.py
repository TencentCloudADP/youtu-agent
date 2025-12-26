import logging

from utu.agents import SimpleAgent
from utu.config import ConfigLoader


async def test_chat_streamed():
    config = ConfigLoader.load_agent_config("test/test_env")
    async with SimpleAgent(config=config) as agent:
        tools = await agent.get_tools()
        logging.info(f"Loaded {len(tools)} tools: {tools}")
        for prompt in [
            "please pwd",
            "please use python to create an empty file `test.txt`",
            "please use `edit_file` to add a line into the test file with one line `test line`",
        ]:
            print(prompt.center(80, "-"))
            await agent.chat_streamed(prompt)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_chat_streamed())
