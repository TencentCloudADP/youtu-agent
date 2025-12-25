"""
- [x] integrate into UI
    use `interaction_toolkit.set_ask_function()` to config ask function
- [x] bug fix: `task_recorder.stream_events()` cannot stop
    ref: `OrchestraAgent`
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from agents import RunResultStreaming, StopAtTools, trace
from agents.function_schema import FuncSchema
from pydantic import BaseModel

from ..agents import SimpleAgent
from ..agents.common import DataClassWithStreamEvents, QueueCompleteSentinel
from ..config import ConfigLoader
from ..tools import TOOLKIT_MAP, UserInteractionToolkit
from ..tools.utils import AgentsMCPUtils, get_tools_schema
from ..utils import DIR_ROOT, FileUtils, get_logger

logger = get_logger(__name__)


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


def add_indented_lines(lines: str | list[str], indent: int = 2) -> str:
    if isinstance(lines, str):
        lines = lines.split("\n")
    return "\n".join(" " * indent + line for line in lines)


class SimpleAgentGenerator:
    def __init__(self, ask_function=None, mode="local"):
        self.prompts = FileUtils.load_prompts("meta/simple_agent_generator.yaml")
        self.output_dir = DIR_ROOT / "configs/agents/generated"
        self.output_dir.mkdir(exist_ok=True)

        self.mode = mode  # local | webui  # NOTE: it is not used now!
        self._initialized = False
        self.ask_function = ask_function
        self.toolkit_name_mapping = {}

    async def build(self):
        if self._initialized:
            return
        self.interaction_toolkit = UserInteractionToolkit()
        if self.ask_function:
            self.interaction_toolkit.set_ask_function(self.ask_function)

        self.agent_1 = SimpleAgent(
            name="clarification_agent",
            instructions=self.prompts["REQUIREMENT_CLARIFICATION_SP"],
            tools=self.interaction_toolkit.get_tools_in_agents(),
            tool_use_behavior=StopAtTools(stop_at_tool_names=["final_answer"]),
        )
        self.agent_2 = SimpleAgent(
            name="tool_selection_agent",
            instructions=self.prompts["TOOL_SELECTION_SP"],
        )
        self.agent_3 = SimpleAgent(
            name="instructions_generation_agent",
            instructions=self.prompts["INSTRUCTIONS_GENERATION_SP"],
        )
        self.agent_4 = SimpleAgent(
            name="name_generation_agent",
            instructions=self.prompts["NAME_GENERATION_SP"],
        )
        self._initialized = True

    # async def run(self, user_input: str):
    #     await self.build()
    #     with trace("simple_agent_generator"):
    #         task_recorder = GeneratorTaskRecorder()
    #         # step 1: generate requirements
    #         await self.step1(task_recorder, user_input)
    #         # step 2: select tools
    #         await self.step2(task_recorder)
    #         # step 3: generate instructions
    #         await self.step3(task_recorder)
    #         # step 4: generate name
    #         await self.step4(task_recorder)
    #         ofn, _ = self.format_config(task_recorder)
    #         print(f"Config saved to {ofn}")

    def run_streamed(self, user_input: str) -> GeneratorTaskRecorder:
        with trace("simple_agent_generator"):
            task_recorder = GeneratorTaskRecorder()
            task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder, user_input))
        return task_recorder

    async def _start_streaming(self, task_recorder: GeneratorTaskRecorder, user_input: str):
        try:
            await self.build()
            await self.step1(task_recorder, user_input)
            await self.step2(task_recorder)
            await self.step3(task_recorder)
            await self.step4(task_recorder)
            ofn, config = self.format_config(task_recorder)
            logger.info(f"Generated config saved to {ofn}")
            event = SimpleAgentGeneratedEvent(filename=str(ofn), config_content=config)
            task_recorder._event_queue.put_nowait(event)
        except Exception as e:
            task_recorder._is_complete = True
            task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
            raise e

        task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
        task_recorder._is_complete = True

    def format_config(self, task_recorder: GeneratorTaskRecorder) -> tuple[str, str]:
        toolkits_includes = []
        toolkits_configs = []
        for toolkit_name, tool_names in task_recorder.selected_tools.items():
            # Use original path (e.g., mcp/memory) instead of sanitized name (mcp_memory)
            toolkits_includes.append(
                f"- /tools/{self.toolkit_name_mapping.get(toolkit_name, toolkit_name)}@toolkits.{toolkit_name}"
            )
            toolkits_configs.append(f"{toolkit_name}: {json.dumps({'activated_tools': tool_names})}")
        config = self.prompts["CONFIG_TEMPLATE"].format(
            agent_name=task_recorder.name,
            instructions=add_indented_lines(task_recorder.instructions, 4),
            toolkits_includes=add_indented_lines(toolkits_includes, 2),
            toolkits_configs=add_indented_lines(toolkits_configs, 2),
        )
        ofn = self.output_dir / f"{task_recorder.name}.yaml"
        ofn.write_text(config, encoding="utf-8")
        return ofn, config

    async def step1(self, task_recorder: GeneratorTaskRecorder, user_input: str) -> None:
        """Generate requirements for the agent."""
        async with self.agent_1 as agent:
            result = agent.run_streamed(user_input)
            await self._process_streamed(result, task_recorder)
            task_recorder.requirements = result.final_output

    async def step2(self, task_recorder: GeneratorTaskRecorder) -> None:
        """Select useful tools from available toolkits. Return: {toolkit_name: [tool_name, ...]}"""
        # NOTE: set available tools here
        available_toolkits = [
            "search",
            "document",
            "image",
            "audio",
            "bash",
            "python_executor",  # builtin
            "mcp/memory",  # mcp
        ]
        tools_schema_by_toolkit: dict[str, dict[str, FuncSchema]] = {}
        for toolkit_name in available_toolkits:
            config = ConfigLoader.load_toolkit_config(toolkit_name)
            match config.mode:
                case "builtin":
                    tools_schema = get_tools_schema(TOOLKIT_MAP[toolkit_name])
                case "mcp":
                    tools_schema = await AgentsMCPUtils.get_mcp_tools_schema(config)
                case _:
                    raise ValueError(f"Unsupported toolkit mode: {config.mode}")
            tools_schema_by_toolkit[toolkit_name] = tools_schema
        tools_descs = []
        tool_to_toolkit_name = {}  # NOTE: toolkit_name should be sanitized, e.g. mcp/memory -> mcp_memory
        for toolkit_name, tools_schema in tools_schema_by_toolkit.items():
            # Sanitize toolkit name for use as dictionary key and YAML identifier
            sanitized_name = toolkit_name.replace("/", "_")
            # Preserve original path for config file references
            self.toolkit_name_mapping[sanitized_name] = toolkit_name
            tools_descs.extend(f"- {tool.name}: {tool.description}" for tool in tools_schema.values())
            tool_to_toolkit_name.update({tool.name: sanitized_name for tool in tools_schema.values()})
        logger.info(f"Available tools: {tool_to_toolkit_name}")
        tools_str = "\n".join(tools_descs)
        query = self.prompts["TOOL_SELECTION_TEMPLATE"].format(
            available_tools=tools_str,
            requirement=task_recorder.requirements,
        )
        async with self.agent_2 as agent:
            result = agent.run_streamed(query)
            await self._process_streamed(result, task_recorder)
            selected_tools = result.final_output
            selected_tool_names = json.loads(selected_tools)
        selected_tools = defaultdict(list)
        for tool_name in selected_tool_names:
            selected_tools[tool_to_toolkit_name[tool_name]].append(tool_name)
        task_recorder.selected_tools = selected_tools

    async def step3(self, task_recorder: GeneratorTaskRecorder) -> None:
        """Generate instructions for the agent."""
        async with self.agent_3 as agent:
            result = agent.run_streamed(task_recorder.requirements)
            await self._process_streamed(result, task_recorder)
            task_recorder.instructions = result.final_output

    async def step4(self, task_recorder: GeneratorTaskRecorder) -> None:
        """Generate name for the agent."""
        async with self.agent_4 as agent:
            result = agent.run_streamed(task_recorder.requirements)
            await self._process_streamed(result, task_recorder)
            name = result.final_output
            if len(name) > 50 or " " in name:
                logger.warning(f"Generated name is too long or contains spaces: {name}")
                name = name[:50].replace(" ", "_")
            task_recorder.name = name

    async def _process_streamed(self, run_result_streaming: RunResultStreaming, task_recorder: GeneratorTaskRecorder):
        async for event in run_result_streaming.stream_events():
            task_recorder._event_queue.put_nowait(event)
