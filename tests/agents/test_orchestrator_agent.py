from utu.agents.orchestrator_agent import OrchestratorAgent
from utu.config import ConfigLoader


async def test_orchestra_agent():
    agent = OrchestratorAgent(config=ConfigLoader.load_agent_config("orchestrator/base"))
    run_result = agent.run_streamed("Introduce the main architectures of CNN")
    async for event in run_result.stream_events():
        print(event)
