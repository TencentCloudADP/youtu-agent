"""Utils to inspect tools
- start a MCP server with all tools (can be checked with @modelcontextprotocol/inspector)
"""

import argparse

from mcp.server.fastmcp import FastMCP

from utu.config import ConfigLoader
from utu.tools import TOOLKIT_MAP


def add_tools(toolkit_names: list[str], mcp: FastMCP) -> None:
    for name in toolkit_names:
        print(f"Loading toolkit: {name}")
        config = ConfigLoader.load_toolkit_config(name)
        toolkit = TOOLKIT_MAP[name](config=config)
        for tool_name, tool in (toolkit.get_tools_map_func()).items():
            mcp.add_tool(tool)
            print(f"Added tool: {tool_name}")


def main(toolkit_names: list[str]) -> None:
    mcp = FastMCP(
        "Tools Test Server",
        host="0.0.0.0",
        port=3005,
    )
    add_tools(toolkit_names, mcp)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkits", type=str, nargs="*", help="List of toolkits to load")
    args = parser.parse_args()
    toolkit_names = args.toolkits or list(TOOLKIT_MAP.keys())
    print(f"Loading toolkits: {toolkit_names}")
    main(toolkit_names)
