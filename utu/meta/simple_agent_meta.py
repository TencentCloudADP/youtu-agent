# ruff: noqa: E501
import json

from agents import StopAtTools, function_tool
from agents.function_schema import FuncSchema
from pydantic import BaseModel, Field

from utu.agents import LLMAgent, SimpleAgent
from utu.config import ConfigLoader, ToolkitConfig
from utu.meta.tool_generator_mcp import ToolGenerator
from utu.tools import TOOLKIT_MAP, AsyncBaseToolkit, register_tool
from utu.tools.utils import get_tools_schema
from utu.utils import (
    DIR_ROOT,
    AgentsMCPUtils,
    AgentsUtils,
    FileUtils,
    PrintUtils,
    StringUtils,
    get_logger,
)

logger = get_logger(__name__)


class ToolSelectionResults(BaseModel):
    class ToolSelection(BaseModel):
        toolkit_name: str = Field(description="The name of the toolkit.")
        tool_name: str = Field(description="The name of the tool.")

    tool_selections: list[ToolSelection] = Field(description="The list of tool selections.")


@function_tool(strict_mode=False)
def select_toolkits(selected_toolkits: list[dict], remark: str) -> list:
    """Select tools based on the given list of tool selections.

    Args:
        selected_toolkits (list[dict[str, any]]): The list of selected toolkits with tool names.
            - Format: [{"toolkit_name": "toolkit_name", "tool_names": ["tool_name1", "tool_name2", ...]}, ...]
            - Set "tool_names" as [] to select all tools from the toolkit.
            - If there is no toolkits that match the query, return empty list.
        remark (str): The rationales for the selection.
            - If no toolkits are selected, give your reason why there are no toolkits that match the query.
            - Otherwise, give your rationales for the selection.
    """
    return {
        "selected_toolkits": selected_toolkits,
        "remark": remark,
    }


class MetaTools(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig | dict | None = None):
        super().__init__(config)
        self.prompts = FileUtils.load_prompts("meta/simple_agent_generator.yaml")
        self.output_dir = DIR_ROOT / "configs/agents/generated"
        self.output_dir.mkdir(exist_ok=True)
        self.toolkit_names = list(TOOLKIT_MAP.keys())

    async def _get_toolkit_schema(self, toolkit_names: list[str]) -> dict[str, dict[str, FuncSchema]]:
        toolkit_name2toolsschema: dict[str, dict[str, FuncSchema]] = {}
        for toolkit_name in toolkit_names:
            config = ConfigLoader.load_toolkit_config(toolkit_name)
            match config.mode:
                case "builtin":
                    tools_schema = get_tools_schema(TOOLKIT_MAP[toolkit_name])
                case "mcp":
                    tools_schema = await AgentsMCPUtils.get_mcp_tools_schema(config)
                case _:
                    raise ValueError(f"Unsupported toolkit mode: {config.mode}")
            toolkit_name2toolsschema[toolkit_name] = tools_schema
        return toolkit_name2toolsschema

    @register_tool
    async def search_toolkit(self, query: str) -> str:
        """Search for built-in toolkit(s) based on the given query.

        Args:
            query (str): The search query to find the toolkit(s), should be targeted.

        Returns:
            str: toolkit (with selected tools) that match the query.
                If no tool_names is specified, all tools will be selected.
        """
        toolkit_name2toolsschema = await self._get_toolkit_schema(self.toolkit_names)
        query = f"Requirement: {query}\nAvailable toolkits: {toolkit_name2toolsschema}"
        tool_selection_agent = LLMAgent(
            name="toolkit_search_agent",
            instructions=self.prompts["META_TOOL_SELECTION_SP"],
            # output_type=ToolSelectionResults,
            tools=[select_toolkits],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["select_toolkits"]),
        )
        result = tool_selection_agent.run_streamed(query)
        await AgentsUtils.print_stream_events(result.stream_events())
        return result.final_output

    @register_tool
    async def create_tool(self, name: str, description: str) -> str:
        """Create a new tool based on the given name and description.

        Args:
            name (str): The name of the tool.
            description (str): The description of the tool.

        Returns:
            str: Result of the tool creation.
        """
        generator = ToolGenerator(auto_debug=False)  # disable autodebug for now
        recorder = generator.run_streamed(description)
        await AgentsUtils.print_stream_events(recorder.stream_events())
        toolkit_info = {
            "toolkit_name": recorder.mcp_toolname,
            "tool_names": [],
        }
        return f"Tool created: {toolkit_info}\n(tool_names is set as empty list because all tools should be selected)"

    @register_tool
    async def ask_user(self, question: str) -> str:
        """Ask the user for clarification.

        Args:
            question (str): The question to ask the user.

        Returns:
            str: The user's response.
        """
        response = await PrintUtils.async_print_input(question)
        return response

    @register_tool
    async def create_agent_config(
        self, agent_name: str, instructions: str, selected_tools: dict[str, list[str]]
    ) -> str:
        """Create a new agent config based on the given agent name, instructions, and selected tools.

        Args:
            agent_name (str): The name of the agent.
            instructions (str): The instructions of the agent.
            selected_tools (dict[str, list[str]]): The selected tools for the agent.
                - The key is the toolkit name, and the value is the list of tool names.
                - If the value is empty list, all tools in the toolkit will be selected.

        Returns:
            str: Result of the agent config creation.
        """
        # sanitize name
        agent_name = agent_name.replace(" ", "_")
        # generate toolkits includes and configs (YAML part)
        toolkits_includes = []
        toolkits_configs = []
        for toolkit_name, tool_names in selected_tools.items():
            # NOTE: toolkit_name should be sanitized, e.g. mcp/memory -> mcp_memory
            sanitized_name = toolkit_name.replace("/", "_")
            toolkits_includes.append(f"- /tools/{toolkit_name}@toolkits.{sanitized_name}")
            toolkits_configs.append(f"{sanitized_name}: {json.dumps({'activated_tools': tool_names})}")
        # generate config
        config = self.prompts["CONFIG_TEMPLATE"].format(
            agent_name=agent_name,
            instructions=StringUtils.indent_lines(instructions, 4),
            toolkits_includes=StringUtils.indent_lines(toolkits_includes, 2),
            toolkits_configs=StringUtils.indent_lines(toolkits_configs, 2),
        )
        ofn = self.output_dir / f"{agent_name}.yaml"
        ofn.write_text(config, encoding="utf-8")
        return f"Agent config created at {ofn}"


async def get_meta_agent() -> SimpleAgent:
    meta_tools = MetaTools()
    meta_agent = SimpleAgent(
        name="meta_agent",
        instructions=meta_tools.prompts["META_AGENT_SP"],
        tools=meta_tools.get_tools_in_agents(),
        tool_use_behavior=StopAtTools(stop_at_tool_names=["create_agent_config"]),
    )
    return meta_agent


if __name__ == "__main__":
    import asyncio

    async def main():
        meta_agent = await get_meta_agent()
        task = await PrintUtils.async_print_input("Enter your agent requirements: ")
        recorder = meta_agent.run_streamed(task)
        await AgentsUtils.print_stream_events(recorder.stream_events())

    async def test_meta_tool():
        meta_tools = MetaTools()
        await meta_tools.search_toolkit("search the internet")

    # asyncio.run(test_meta_tool())
    asyncio.run(main())
