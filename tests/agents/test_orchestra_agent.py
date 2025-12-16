import pytest

from utu.agents import OrchestraAgent
from utu.config import ConfigLoader


@pytest.fixture
async def agent():
    agent = OrchestraAgent(config=ConfigLoader.load_agent_config("orchestra/base"))
    await agent.build()
    yield agent


async def test_run_streamed(agent: OrchestraAgent):
    run_result = agent.run_streamed("Introduce the main architectures of CNN")
    async for event in run_result.stream_events():
        print(event)
