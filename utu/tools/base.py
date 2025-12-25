from collections.abc import Callable
from typing import TYPE_CHECKING

import mcp.types as types
from agents import FunctionTool, function_tool

from ..config import ToolkitConfig
from ..utils import ChatCompletionConverter, FileUtils, get_logger
from .utils import MCPConverter, register_tool as register_tool

if TYPE_CHECKING:
    from e2b_code_interpreter import AsyncSandbox

    from ..env import E2BEnv

logger = get_logger(__name__)


class AsyncBaseToolkit:
    """Base class for toolkits."""

    def __init__(self, config: ToolkitConfig | dict | None = None):
        if not isinstance(config, ToolkitConfig):
            config = config or {}
            config = ToolkitConfig(config=config, name=self.__class__.__name__)

        self.config: ToolkitConfig = config
        self._tools_map: dict[str, Callable] = None

        self.env_mode = self.config.env_mode
        if self.env_mode == "e2b":
            self.e2b_env: E2BEnv = None
            self.e2b_sandbox: AsyncSandbox = None

    def setup_e2b_env(self, env: "E2BEnv") -> None:
        if self.env_mode != "e2b":
            logger.warning(f"Toolkit should not setup e2b sandbox in env_mode {self.env_mode}!")
            return
        self.e2b_env = env
        self.e2b_sandbox = env.sandbox

    @property
    def tools_map(self) -> dict[str, Callable]:
        """Lazy loading of tools map.
        - collect tools registered by @register_tool
        """
        if self._tools_map is None:
            self._tools_map = {}
            # Iterate through all methods of the class and register @tool
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if callable(attr) and getattr(attr, "_is_tool", False):
                    self._tools_map[attr._tool_name] = attr
        return self._tools_map

    def get_tools_map_func(self) -> dict[str, Callable]:
        """Get tools map. It will filter tools by config.activated_tools if it is not None."""
        if self.config.activated_tools:
            assert all(tool_name in self.tools_map for tool_name in self.config.activated_tools), (
                f"Error config activated tools: {self.config.activated_tools}! available tools: {self.tools_map.keys()}"
            )
            tools_map = {tool_name: self.tools_map[tool_name] for tool_name in self.config.activated_tools}
        else:
            tools_map = self.tools_map
        return tools_map

    def get_tools_in_agents(self) -> list[FunctionTool]:
        """Get tools in openai-agents format."""
        tools_map = self.get_tools_map_func()
        tools = []
        for _, tool in tools_map.items():
            tools.append(
                function_tool(
                    tool,
                    strict_mode=False,  # turn off strict mode
                )
            )
        return tools

    def get_tools_in_openai(self) -> list[dict]:
        """Get tools in OpenAI format."""
        tools = self.get_tools_in_agents()
        return [ChatCompletionConverter.tool_to_openai(tool) for tool in tools]

    def get_tools_in_mcp(self) -> list[types.Tool]:
        """Get tools in MCP format."""
        tools = self.get_tools_in_agents()
        return [MCPConverter.function_tool_to_mcp(tool) for tool in tools]

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool by its name."""
        tools_map = self.get_tools_map_func()
        if name not in tools_map:
            raise ValueError(f"Tool {name} not found")
        tool = tools_map[name]
        return await tool(**arguments)


TOOL_PROMPTS: dict[str, str] = FileUtils.load_prompts("tools/tools_prompts.yaml")
