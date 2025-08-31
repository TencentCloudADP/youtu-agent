import json
import logging

from agents import FunctionTool, RunContextWrapper, TContext, Tool

from .base_env import BaseEnv
from .utils import DockerManager, MCPClient

logger = logging.getLogger(__name__)


class BrowserEnv(BaseEnv):
    """Browser environment for agents."""

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.docker_manager = DockerManager()
        self.browser_state: str = None

    async def build(self):
        """Build the environment. We use docker to run a browser container."""
        self.container_info = await self.docker_manager.start_container(self.trace_id)
        self.mcp_url = self.container_info["mcp_url"]

    async def cleanup(self):
        await self.docker_manager.stop_container(self.trace_id)

    def get_state(self) -> str:
        """Get the current state of the environment."""
        return self.browser_state

    async def get_tools(self) -> list[Tool]:
        """Get the tools available in the environment."""
        activated_tools = (
            "search_google",
            "go_to_url",
            "go_back",
            # "wait",
            "click_element",
            "input_text",
            "switch_tab",
            "open_tab",
            "scroll_down",
            "scroll_up",
            "download_file",
            # "search_google_api"
        )
        tools: list[Tool] = []

        def create_on_invoke_tool(tool_name: str):
            async def on_invoke_tool(ctx: RunContextWrapper[TContext], input_json: str) -> str:
                try:
                    async with MCPClient.get_mcp_client(self.mcp_url) as client:
                        res = await client.call_tool(tool_name, json.loads(input_json))
                        if res.isError:
                            return f"Error: {res.content[0].text}"
                        self.browser_state = res.content[1].text  # DISCUSS: record the web actions?
                        return res.content[0].text
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(f"except: {e}", exc_info=True)
                    return f"Error: {e}"

            return on_invoke_tool

        tool_defs = []
        async with MCPClient.get_mcp_client(self.mcp_url) as client:
            # NOTE: check `MCPUtil` in @agents
            res = await client.list_tools()
            assert res.nextCursor is None
            tool_defs.extend(res.tools)

        for tool_def in tool_defs:
            if tool_def.name not in activated_tools:
                continue
            tools.append(
                FunctionTool(
                    name=tool_def.name,
                    description=tool_def.description,
                    params_json_schema=tool_def.inputSchema,
                    on_invoke_tool=create_on_invoke_tool(tool_def.name),
                )
            )
        return tools
