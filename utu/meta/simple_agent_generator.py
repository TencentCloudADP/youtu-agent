"""
- [x] integrate into UI
    use `interaction_toolkit.set_ask_function()` to config ask function
- [ ] purify the phoenix tracing, ref @training-free GRPO
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from agents import RunResultStreaming, StopAtTools, trace
from agents.function_schema import FuncSchema
from pydantic import BaseModel

from ..agents import LLMAgent, SimpleAgent
from ..agents.common import DataClassWithStreamEvents, QueueCompleteSentinel
from ..config import ConfigLoader
from ..tools import TOOLKIT_MAP, UserInteractionToolkit
from ..tools.utils import AgentsMCPUtils, get_tools_schema
from ..utils import DIR_ROOT, AgentsUtils, FileUtils, LLMOutputParser, StringUtils, get_logger
from .tool_generator_mcp import ToolGenerator

logger = get_logger(__name__)
DEFAULT_AVAILABLE_TOOLS = [
    "search",
    "document",
    "image",
    "audio",
    "bash",
    "python_executor",
    "mcp/memory",  # mcp
]


@dataclass
class GeneratorTaskRecorder(DataClassWithStreamEvents):
    requirements: str = field(default=None)
    selected_tools: dict[str, list[str]] = field(default=None)
    instructions: str = field(default=None)
    name: str = field(default=None)


class SimpleAgentGeneratedEvent(BaseModel):
    type: Literal["simple_agent_generated"] = "simple_agent_generated"
    config_content: str
    filename: str


class SimpleAgentGeneratorConfig(BaseModel):
    available_toolkits: list[str] = DEFAULT_AVAILABLE_TOOLS


class SimpleAgentGenerator:
    def __init__(self, ask_function=None, config: SimpleAgentGeneratorConfig = None):
        self.config = config or SimpleAgentGeneratorConfig()
        self.prompts = FileUtils.load_prompts("meta/simple_agent_generator.yaml")
        self.output_dir = DIR_ROOT / "configs/agents/generated"
        self.output_dir.mkdir(exist_ok=True)

        self.interaction_toolkit = UserInteractionToolkit()
        if ask_function:
            self.interaction_toolkit.set_ask_function(ask_function)

    # only support streamed mode
    def run_streamed(self, user_input: str) -> GeneratorTaskRecorder:
        with trace("simple_agent_generator"):
            task_recorder = GeneratorTaskRecorder()
            task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder, user_input))
        return task_recorder

    async def _start_streaming(self, task_recorder: GeneratorTaskRecorder, user_input: str):
        try:
            await self.step1(task_recorder, user_input)
            await self.step2(task_recorder)
            await self.step3(task_recorder)
            await self.step4(task_recorder)
            ofn, config = self.format_config(task_recorder)
            logger.info(f"Generated config saved to {ofn}")
            event = SimpleAgentGeneratedEvent(filename=str(ofn), config_content=config)
            task_recorder._event_queue.put_nowait(event)
            # mark complete
            task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
            task_recorder._is_complete = True
        except Exception as e:
            task_recorder._is_complete = True
            task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
            raise e

    def format_config(self, task_recorder: GeneratorTaskRecorder) -> tuple[str, str]:
        toolkits_includes = []
        toolkits_configs = []
        for toolkit_name, tool_names in task_recorder.selected_tools.items():
            # NOTE: toolkit_name should be sanitized, e.g. mcp/memory -> mcp_memory
            sanitized_name = toolkit_name.replace("/", "_")
            toolkits_includes.append(f"- /tools/{toolkit_name}@toolkits.{sanitized_name}")
            toolkits_configs.append(f"{sanitized_name}: {json.dumps({'activated_tools': tool_names})}")
        config = self.prompts["CONFIG_TEMPLATE"].format(
            agent_name=task_recorder.name,
            instructions=StringUtils.indent_lines(task_recorder.instructions, 4),
            toolkits_includes=StringUtils.indent_lines(toolkits_includes, 2),
            toolkits_configs=StringUtils.indent_lines(toolkits_configs, 2),
        )
        ofn = self.output_dir / f"{task_recorder.name}.yaml"
        ofn.write_text(config, encoding="utf-8")
        return ofn, config

    async def step1(self, task_recorder: GeneratorTaskRecorder, user_input: str) -> None:
        """Generate requirements for the agent."""
        # TODO: purify this step
        clarification_agent = SimpleAgent(
            name="clarification_agent",
            instructions=self.prompts["REQUIREMENT_CLARIFICATION_SP"],
            tools=self.interaction_toolkit.get_tools_in_agents(),
            tool_use_behavior=StopAtTools(stop_at_tool_names=["final_answer"]),
        )
        async with clarification_agent as agent:
            result = agent.run_streamed(user_input)
            await self._process_streamed(result, task_recorder)
            task_recorder.requirements = result.final_output

    async def step2(self, task_recorder: GeneratorTaskRecorder) -> None:
        """Select useful tools from available toolkits."""
        # 1. generate new tools if needed
        toolkit_name2toolsschema = await self._get_toolkit_schema(self.config.available_toolkits)
        tool_generation_agent = LLMAgent(name="tool_generation_agent", instructions=self.prompts["TOOL_GENERATION_SP"])
        q = self.prompts["TOOL_GENERATION_TEMPLATE"].format(
            requirement=task_recorder.requirements, available_tools=self._format_tools_str(toolkit_name2toolsschema)
        )
        res = tool_generation_agent.run_streamed(q)
        await self._process_streamed(res, task_recorder)
        new_tools = LLMOutputParser.extract_code_json(res.final_output).get("new_tools", [])
        if new_tools:
            new_mcp_toolnames = []
            generator = ToolGenerator(auto_debug=False)  # disable autodebug for now
            for new_tool in new_tools:
                if "description" not in new_tool:
                    logger.warning(f"Tool description missing for tool: {new_tool}, skipping...")
                    continue
                logger.info(f"Generated new tool: {new_tool}")
                recorder = generator.run_streamed(new_tool["description"])
                await AgentsUtils.print_stream_events(recorder.stream_events())
                new_mcp_toolnames.append(recorder.mcp_toolname)
            # add the generated tools to available tools
            if new_mcp_toolnames:
                toolkit_name2toolsschema |= await self._get_toolkit_schema(new_mcp_toolnames)

        # 2. format available tools
        tool_to_toolkit_name = {}
        for toolkit_name, tools_schema in toolkit_name2toolsschema.items():
            tool_to_toolkit_name.update({tool.name: toolkit_name for tool in tools_schema.values()})
        logger.info(f"Available tools: {tool_to_toolkit_name}")
        query = self.prompts["TOOL_SELECTION_TEMPLATE"].format(
            available_tools=self._format_tools_str(toolkit_name2toolsschema),
            requirement=task_recorder.requirements,
        )
        # 3. run agent to select tools
        tool_selection_agent = LLMAgent(name="tool_selection_agent", instructions=self.prompts["TOOL_SELECTION_SP"])
        result = tool_selection_agent.run_streamed(query)
        await self._process_streamed(result, task_recorder)
        selected_tools = defaultdict(list)
        for tool_name in LLMOutputParser.extract_code_json(result.final_output):
            selected_tools[tool_to_toolkit_name[tool_name]].append(tool_name)
        task_recorder.selected_tools = selected_tools

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

    def _format_tools_str(self, toolkit_name2toolsschema: dict[str, dict[str, FuncSchema]]) -> str:
        tools_descs = []
        for _, tools_schema in toolkit_name2toolsschema.items():
            for tool in tools_schema.values():
                tools_descs.append(f"- {tool.name}: {StringUtils.remove_newlines(tool.description)}")
        return "\n".join(tools_descs)

    async def step3(self, task_recorder: GeneratorTaskRecorder) -> None:
        """Generate instructions for the agent."""
        instructions_generation_agent = LLMAgent(
            name="instructions_generation_agent", instructions=self.prompts["INSTRUCTIONS_GENERATION_SP"]
        )
        result = instructions_generation_agent.run_streamed(task_recorder.requirements)
        await self._process_streamed(result, task_recorder)
        task_recorder.instructions = result.final_output

    async def step4(self, task_recorder: GeneratorTaskRecorder) -> None:
        """Generate name for the agent."""
        name_generation_agent = LLMAgent(
            name="name_generation_agent",
            instructions=self.prompts["NAME_GENERATION_SP"],
        )
        result = name_generation_agent.run_streamed(task_recorder.requirements)
        await self._process_streamed(result, task_recorder)
        name = result.final_output
        if len(name) > 50 or " " in name:
            logger.warning(f"Generated name is too long or contains spaces: {name}")
            name = name[:50].replace(" ", "_")
        task_recorder.name = name

    async def _process_streamed(self, run_result_streaming: RunResultStreaming, task_recorder: GeneratorTaskRecorder):
        async for event in run_result_streaming.stream_events():
            task_recorder._event_queue.put_nowait(event)
