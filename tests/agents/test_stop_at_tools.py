from utu.agents import SimpleAgent
from utu.config import ConfigLoader


async def main():
    config = ConfigLoader.load_agent_config("simple/base_search")
    config.stop_at_tool_names = ["search"]
    async with SimpleAgent(config=config) as agent:
        res = await agent.chat_streamed("Search for Gemini 3.")
        print(f"res.final_output: {res.final_output}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
