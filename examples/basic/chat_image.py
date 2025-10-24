"""A basic example of chatting with image input.

Note the LLM model should support image input, e.g., gpt-4o or glm-4.5v
"""

import asyncio

from utu.agents import get_agent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils, PrintUtils


async def main():
    image_url = "https://lightblues.github.io/img/CowboyBepop/cb-1.jpg"
    agent = get_agent(ConfigLoader.load_agent_config("simple/base"))

    turn_id = 0
    while True:
        user_input = await PrintUtils.async_print_input("> ")
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            break
        if not user_input.strip():
            continue
        if turn_id == 0:
            # injected image into the first turn", color="gray
            user_input = [AgentsUtils.get_message_from_image(image_url)] + [{"role": "user", "content": user_input}]
        await agent.chat_streamed(user_input)
        turn_id += 1


if __name__ == "__main__":
    asyncio.run(main())
