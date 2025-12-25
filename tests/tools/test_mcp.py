from utu.config import ConfigLoader
from utu.tools.utils import AgentsMCPUtils


async def test_mcp():
    # config_name = "mcp/time"
    # action = [("get_current_time", {"timezone": "Asia/Shanghai"})]
    # config_name = "mcp/sequentialthinking"
    # action = [("sequentialthinking", {"thought": "step 1", "nextThoughtNeeded": False, "thoughtNumber": 1, "totalThoughts": 1})]  # ruff: noqa: E501

    # config_name = "mcp/context7"
    # action = [("get-library-docs", {"context7CompatibleLibraryID": "/mongodb/docs"})]
    config_name = "mcp/github"
    action = [("search_repositories", {"query": "youtu-agent"})]

    # config_name = "mcp/mem0"
    # action = [("add_memories", {"text": "I like to eat bagels"}), ("list_memories", {})]

    config = ConfigLoader.load_toolkit_config(config_name)
    mcp_server = AgentsMCPUtils.get_mcp_server(config)
    async with mcp_server:
        tools = await mcp_server.list_tools()
        print("Tools:", tools)
        for name, params in action:
            print(f"Calling tool: {name} with params: {params}")
            res = await mcp_server.call_tool(name, params)
            print(f"> res: {res}")


async def test_get_mcp_tools_schema():
    config_name = "mcp/github"
    config = ConfigLoader.load_toolkit_config(config_name)
    tools_map = await AgentsMCPUtils.get_mcp_tools_schema(config)
    for name, schema in tools_map.items():
        print(f"Tool: {name}, schema: {schema}")
