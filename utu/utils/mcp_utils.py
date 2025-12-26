from typing import TYPE_CHECKING

import mcp.types as types
from agents import Agent, FunctionTool, RunContextWrapper, Tool
from agents.function_schema import FuncSchema
from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio, MCPServerStreamableHttp, MCPUtil, ToolFilterStatic
from mcp import Tool as MCPTool

if TYPE_CHECKING:
    from ..config import ToolkitConfig


MCP_SERVER_MAP = {
    "sse": MCPServerSse,
    "stdio": MCPServerStdio,
    "streamable_http": MCPServerStreamableHttp,
}


class AgentsMCPUtils:
    @classmethod
    def get_mcp_server(cls, config: "ToolkitConfig") -> MCPServerSse | MCPServerStdio | MCPServerStreamableHttp:
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
    async def get_tools_mcp(cls, config: "ToolkitConfig") -> list[MCPTool]:
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
    async def get_mcp_tools_schema(cls, config: "ToolkitConfig") -> dict[str, FuncSchema]:
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
