import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import mcp.types as types
from agents import Agent, FunctionTool, RunContextWrapper, Tool
from agents.function_schema import FuncSchema, function_schema
from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio, MCPServerStreamableHttp, MCPUtil, ToolFilterStatic
from mcp import Tool as MCPTool

from ..config import ToolkitConfig

if TYPE_CHECKING:
    from e2b.sandbox.commands.command_handle import CommandExitException, CommandResult
    from e2b_code_interpreter.models import Execution


# ------------------------------------------------------------------------------
# e2b
class E2BUtils:
    @classmethod
    def execution_to_str(cls, execution: "Execution") -> str:
        """Convert e2b Execution to string.
        The official .to_json() is not good for Chinese output!"""
        from e2b_code_interpreter.models import serialize_results

        logs = execution.logs
        logs_data = {"stdout": logs.stdout, "stderr": logs.stderr}
        error = execution.error
        error_data = {"name": error.name, "value": error.value, "traceback": error.traceback} if error else None
        result = {
            "result": serialize_results(execution.results),
            "logs": logs_data,  # execution.logs.to_json(),
            "error": error_data,  # execution.error.to_json() if execution.error else None
        }
        return json.dumps(result, ensure_ascii=False)

    @classmethod
    def command_result_to_str(cls, command_result: "CommandResult") -> str:
        """Convert e2b CommandResult to string."""
        result = {
            "stdout": command_result.stdout,
            "stderr": command_result.stderr,
            "exit_code": command_result.exit_code,
            "error": command_result.error,
        }
        return json.dumps(result, ensure_ascii=False)

    @classmethod
    def command_exit_exception_to_str(cls, command_exception: "CommandExitException") -> str:
        result = {
            "stdout": command_exception.stdout,
            "stderr": command_exception.stderr,
            "exit_code": command_exception.exit_code,
            "error": command_exception.error,
        }
        return json.dumps(result, ensure_ascii=False)


# ------------------------------------------------------------------------------
# MCP
MCP_SERVER_MAP = {
    "sse": MCPServerSse,
    "stdio": MCPServerStdio,
    "streamable_http": MCPServerStreamableHttp,
}


class AgentsMCPUtils:
    @classmethod
    def get_mcp_server(cls, config: ToolkitConfig) -> MCPServerSse | MCPServerStdio | MCPServerStreamableHttp:
        """Get mcp server from config, with tool_filter if activated_tools is set.
        NOTE: you should manage the lifecycle of the returned server (.connect & .cleanup), e.g. using `async with`."""
        assert config.mode == "mcp", f"config mode must be 'mcp', got {config.mode}"
        assert config.mcp_transport in MCP_SERVER_MAP, f"Unsupported mcp transport: {config.mcp_transport}"
        tool_filter = ToolFilterStatic(allowed_tool_names=config.activated_tools) if config.activated_tools else None
        return MCP_SERVER_MAP[config.mcp_transport](
            params=config.config,
            name=config.name,
            client_session_timeout_seconds=config.mcp_client_session_timeout_seconds,
            tool_filter=tool_filter,
        )

    @classmethod
    async def get_tools_mcp(cls, config: ToolkitConfig) -> list[MCPTool]:
        async with cls.get_mcp_server(config) as mcp_server:
            # It is required to pass agent and run_context when using `tool_filter`, we pass a dummy agent here
            tools = await mcp_server.list_tools(run_context=RunContextWrapper(context=None), agent=Agent(name="dummy"))
            return tools

    @classmethod
    async def get_tools_agents(cls, mcp_server: MCPServer) -> list[Tool]:
        return await MCPUtil.get_function_tools(
            mcp_server,
            convert_schemas_to_strict=False,
            run_context=RunContextWrapper(context=None),
            agent=Agent(name="dummy"),
        )

    @classmethod
    async def get_mcp_tools_schema(cls, config: ToolkitConfig) -> dict[str, FuncSchema]:
        """Get MCP tools schema from config."""
        tools = await cls.get_tools_mcp(config)
        tools_map = {}
        for tool in tools:
            tools_map[tool.name] = FuncSchema(
                name=tool.name,
                description=tool.description,
                params_pydantic_model=None,
                params_json_schema=tool.inputSchema,
                signature=None,
            )
        return tools_map


class MCPConverter:
    @classmethod
    def function_tool_to_mcp(cls, tool: FunctionTool) -> types.Tool:
        return types.Tool(
            name=tool.name,
            description=tool.description,
            inputSchema=tool.params_json_schema,
        )


# ------------------------------------------------------------------------------
# AsyncBaseToolkit utils
def register_tool(name: str = None):
    """Decorator to register a method as a tool.

    Usage:
        @register_tool  # uses method name
        @register_tool()  # uses method name
        @register_tool("custom_name")  # uses custom name

    Args:
        name (str, optional): The name of the tool. (Also support passing the function)
    """

    def decorator(func: Callable):
        if isinstance(name, str):
            tool_name = name
        else:
            tool_name = func.__name__
        func._is_tool = True
        func._tool_name = tool_name
        return func

    if callable(name):
        return decorator(name)
    return decorator


def get_tools_map(cls: type) -> dict[str, Callable]:
    """Get tools map from a class, without instance the class."""
    tools_map = {}
    # Iterate through all methods of the class and register @tool
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and getattr(attr, "_is_tool", False):
            tools_map[attr._tool_name] = attr
    return tools_map


def get_tools_schema(cls: type) -> dict[str, FuncSchema]:
    """Get tools schema from a class, without instance the class."""
    tools_map = {}
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and getattr(attr, "_is_tool", False):
            tools_map[attr._tool_name] = function_schema(attr)
    return tools_map


# ------------------------------------------------------------------------------
# misc
class ContentFilter:
    def __init__(self, banned_sites: list[str] = None):
        if banned_sites:
            self.RE_MATCHED_SITES = re.compile(r"^(" + "|".join(banned_sites) + r")")
        else:
            self.RE_MATCHED_SITES = None

    def filter_results(self, results: list[dict], limit: int, key: str = "link") -> list[dict]:
        # can also use search operator `-site:huggingface.co`
        # ret: {title, link, snippet, position, | sitelinks}
        res = []
        for result in results:
            if self.RE_MATCHED_SITES is None or not self.RE_MATCHED_SITES.match(result[key]):
                res.append(result)
            if len(res) >= limit:
                break
        return res
