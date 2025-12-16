from agents import Agent, Runner

from utu.config import ConfigLoader
from utu.models.react import ReactModel, get_react_model
from utu.tools import SearchToolkit
from utu.utils import AgentsUtils


async def test_react_model():
    model_config = ConfigLoader.load_model_config()
    model: ReactModel = get_react_model(**model_config.model_provider.model_dump())
    agent = Agent(
        name="test_agent",
        instructions="You are a helpful assistant.",
        model=model,
        tools=SearchToolkit().get_tools_in_agents(),
        mcp_servers=[],
    )

    result = Runner.run_streamed(agent, "Tell me about Google")
    await AgentsUtils.print_stream_events(result.stream_events())
    # result = await Runner.run(agent, "Tell me about Google")
    # AgentsUtils.print_new_items(result.new_items)
    next_input_items = result.to_input_list()
    print(next_input_items)
