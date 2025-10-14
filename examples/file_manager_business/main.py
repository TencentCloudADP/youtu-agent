import asyncio
import pathlib

from utu.agents import SimpleAgent
from utu.config import ConfigLoader
from utu.agents import SimpleAgent

EXAMPLE_QUERY = (
    "按照文件类型归档当前文件夹下的文件，并且统一文件名的风格。",
    "要求：对于不同类型的文件，创建不同的文件夹。在各自类型的文件夹下，按照合同方再次进行归档。对于发票，可按照大小进行归档，分为大额和小额。",
    "如果存在某个文件有多个版本，比如说后缀v1等等，需要将所有版本的文件都归档到同一个文件夹下。",
    "临时文件、无关文件，直接清理到一个统一的文件夹中，然后不再管他。",
    "建议：先ls检查当前目录，对每个文件都想好要做的操作之后，在一次bash工具调用中完成多个操作。",
    "所有操作都完成后，再检查当前目录，看是否还有文件需要处理。"
    "提示：所有文件都是文本文件，可以通过cat读出。"
)


config = ConfigLoader.load_agent_config("examples/file_manager_business")
worker_agent = SimpleAgent(config=config)


async def main():
    async with worker_agent as agent:
        result = await agent.chat_streamed(EXAMPLE_QUERY)

if __name__ == "__main__":
    asyncio.run(main())
