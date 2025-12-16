import json
from contextlib import AsyncExitStack

from agents import FunctionTool, RunContextWrapper, TContext, Tool

from ..utils import get_logger
from .base_env import BaseEnv
from .utils import MCPClient
from .utils.tione_manager import TioneEnvManager

logger = get_logger(__name__)


class BrowserTioneEnv(BaseEnv):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.browser_state: str = None
        self._stack = AsyncExitStack()

    async def build(self):
        """Build the environment. We use docker to run a browser container."""
        tione_manager = TioneEnvManager(type="BROWSER_CHROMIUM")
        logger.info(">> entering browser context")
        env_info = await tione_manager.create_env()
        self.mcp_url = f"http://{env_info['Endpoint']}/mcp/"
        logger.debug(">> entering browser context")
        self.client = await self._stack.enter_async_context(MCPClient.get_mcp_client(self.mcp_url))

    async def cleanup(self):
        logger.debug(">> exiting browser context")
        await self._stack.aclose()

    def get_state(self) -> str:
        """Get the current state of the environment."""
        logger.debug(f">> injected state {str(self.browser_state)[:100]}")
        return self.browser_state

    async def get_tools(self) -> list[Tool]:
        """Get the tools available in the environment."""

        def create_on_invoke_tool(tool_name: str):
            async def on_invoke_tool(ctx: RunContextWrapper[TContext], input_json: str) -> str:
                try:
                    res = await self.client.call_tool(tool_name, json.loads(input_json))
                    if res.isError:
                        return f"Error: {res.content[0].text}"
                    if self.config.get("state_keep", True):
                        # state_keep=True: keep all the browser states in history messages
                        tool_result = res.content[0].text + res.content[1].text
                    else:
                        # state_keep=False: only keep the latest browser state
                        self.browser_state = res.content[1].text
                        tool_result = res.content[0].text
                    return tool_result
                except Exception as e:
                    logger.error(f"except: {e}", exc_info=True)
                    return f"Error: {e}"

            return on_invoke_tool

        # NOTE: check `MCPUtil` in @agents
        res = await self.client.list_tools()
        assert res.nextCursor is None
        tools = []
        for tool in res.tools:
            tools.append(
                FunctionTool(
                    name=tool.name,
                    description=tool.description,
                    params_json_schema=tool.inputSchema,
                    on_invoke_tool=create_on_invoke_tool(tool.name),
                )
            )
        return tools
