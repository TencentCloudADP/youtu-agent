from pathlib import Path

import pytest
from agents import FunctionTool, Model, function_tool
from agents.tool_context import ToolContext

from utu.agents import SimpleAgent
from utu.config import AgentConfig, ToolkitConfig
from utu.tools.base import AsyncBaseToolkit, register_tool


class NoModel(Model):
    async def get_response(self, *args, **kwargs):
        raise AssertionError("Model calls are not needed for tool lifecycle tests")

    async def stream_response(self, *args, **kwargs):
        raise AssertionError("Model calls are not needed for tool lifecycle tests")
        yield


class FileToolkit(AsyncBaseToolkit):
    async def build(self):
        self.handle = open(self.config.config["path"], encoding="utf-8")

    async def cleanup(self):
        self.handle.close()

    @register_tool
    async def read_resource(self) -> str:
        """Read the configured local text resource."""
        self.handle.seek(0)
        return self.handle.read()


@pytest.fixture
def resource_config(tmp_path):
    resource = tmp_path / "resource.txt"
    resource.write_text("resource available", encoding="utf-8")
    return AgentConfig(
        toolkits={
            "resource": ToolkitConfig(
                name="resource",
                mode="customized",
                customized_filepath=str(Path(__file__).resolve()),
                customized_classname="FileToolkit",
                config={"path": str(resource)},
            )
        }
    )


async def invoke(tool: FunctionTool) -> str:
    context = ToolContext(context={}, tool_name=tool.name, tool_call_id="call_resource", tool_arguments="{}")
    return await tool.on_invoke_tool(context, "{}")


@pytest.mark.parametrize("tools", [None, []])
async def test_rebuild_reopens_configured_toolkit(resource_config, tools):
    agent = SimpleAgent(config=resource_config, model=NoModel(), tools=tools)
    previous_tool = None
    previous_toolkit = None
    for _ in range(3):
        async with agent:
            tool = agent.current_agent.tools[0]
            assert await invoke(tool) == "resource available"
            toolkit = agent._toolkits["resource"]
            assert tool is not previous_tool
            assert toolkit is not previous_toolkit
            await agent.build()
            assert agent.current_agent.tools[0] is tool
            assert agent._toolkits["resource"] is toolkit
            previous_tool = tool
            previous_toolkit = toolkit
        assert toolkit.handle.closed
        assert agent._toolkits == {}


async def test_rebuild_preserves_provided_tools(resource_config):
    @function_tool
    async def caller_tool() -> str:
        """Return a caller-owned value."""
        return "caller value"

    @function_tool
    async def additional_tool() -> str:
        """Return another caller-owned value."""
        return "additional value"

    tools = [caller_tool]
    agent = SimpleAgent(config=resource_config, model=NoModel(), tools=tools)
    async with agent:
        assert agent.tools is tools
        assert agent.current_agent.tools is tools
        assert agent.current_agent.tools[0] is caller_tool
        assert await invoke(caller_tool) == "caller value"
        assert agent._toolkits == {}
    assert agent.tools is tools
    assert await invoke(caller_tool) == "caller value"
    tools.append(additional_tool)
    async with agent:
        assert agent.tools is tools
        assert agent.current_agent.tools is tools
        assert agent.current_agent.tools[0] is caller_tool
        assert agent.current_agent.tools[1] is additional_tool
        assert await invoke(additional_tool) == "additional value"
        assert agent._toolkits == {}


@pytest.mark.parametrize("tools", [None, []])
async def test_rebuild_without_tools(tools):
    agent = SimpleAgent(config=AgentConfig(), model=NoModel(), tools=tools)
    for _ in range(2):
        async with agent:
            assert agent.current_agent.tools == []
            assert agent._toolkits == {}
