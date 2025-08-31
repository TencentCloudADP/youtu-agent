from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, Literal

from agents import (
    Agent,
    AgentOutputSchemaBase,
    Model,
    ModelSettings,
    RunConfig,
    RunHooks,
    Runner,
    RunResult,
    RunResultStreaming,
    StopAtTools,
    TContext,
    Tool,
    TResponseInputItem,
    trace,
)
from agents.mcp import MCPServer, MCPServerStdio

from ..config import AgentConfig, ConfigLoader, ToolkitConfig
from ..context import BaseContextManager, build_context_manager
from ..env import BaseEnv, get_env
from ..tools import TOOLKIT_MAP, AsyncBaseToolkit
from ..utils import AgentsUtils, get_logger
from .base_agent import BaseAgent
from .common import TaskRecorder

logger = get_logger(__name__)


class SimpleAgent(BaseAgent):
    """A simple agent with env, tools, mcps, and context manager, wrapped on openai-agents."""

    def __init__(
        self,
        *,
        config: AgentConfig | str | None = None,  # use config to pass agent configs
        name: str | None = None,
        instructions: str | Callable | None = None,
        model: str | Model | None = None,
        model_settings: ModelSettings | None = None,
        tools: list[Tool] = None,
        output_type: type[Any] | AgentOutputSchemaBase | None = None,
        tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools = "run_llm_again",
    ):
        self.config = self._get_config(config)
        if name:
            self.config.agent.name = name
        if instructions:
            self.config.agent.instructions = instructions
        self.model = self._get_model(self.config, model)
        self.model_settings = self._get_model_settings(self.config, model_settings)
        self.tools: list[Tool] = tools or []
        self.output_type: type[Any] | AgentOutputSchemaBase | None = output_type
        self.tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools = tool_use_behavior
        self.context_manager: BaseContextManager = None
        self.env: BaseEnv = None
        self.current_agent: Agent[TContext] = None  # move to task recorder?
        self.input_items: list[TResponseInputItem] = []

        self._run_hooks: RunHooks = None
        self._mcp_servers: list[MCPServer] = []
        self._toolkits: list[AsyncBaseToolkit] = []
        self._mcps_exit_stack = AsyncExitStack()
        self._tools_exit_stack = AsyncExitStack()

    def _get_config(self, config: AgentConfig | str | None) -> AgentConfig:
        if isinstance(config, AgentConfig):
            return config
        return ConfigLoader.load_agent_config(config or "base")

    def _get_model(self, config: AgentConfig, model: str | Model | None = None) -> Model:
        if isinstance(model, Model):
            return model
        model_provider_config = config.model.model_provider.model_dump()
        if isinstance(model, str):
            model_provider_config["model"] = model
        return AgentsUtils.get_agents_model(**model_provider_config)

    def _get_model_settings(self, config: AgentConfig, model_settings: ModelSettings | None = None) -> ModelSettings:
        if isinstance(model_settings, ModelSettings):
            return model_settings
        return config.model.model_settings

    async def __aenter__(self):
        await self.build()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def build(self):
        """Build the agent"""
        self.env = await get_env(self.config, "None")  # FIXME: trace_id
        await self.env.build()
        self.current_agent = Agent(
            name=self.config.agent.name,
            instructions=self.config.agent.instructions,
            model=self.model,
            model_settings=self.model_settings,
            tools=await self.get_tools(),
            output_type=self.output_type,
            tool_use_behavior=self.tool_use_behavior,
            mcp_servers=self._mcp_servers,
        )
        self.context_manager = build_context_manager(self.config)

    async def cleanup(self):
        """Cleanup"""
        logger.info("Cleaning up MCP servers...")
        await self._mcps_exit_stack.aclose()
        self._mcp_servers = []
        logger.info("Cleaning up tools...")
        await self._tools_exit_stack.aclose()
        self._toolkits = []
        logger.info("Cleaning up env...")
        await self.env.cleanup()

    async def get_tools(self) -> list[Tool]:
        if self.tools:
            return self.tools

        tools_list: list[Tool] = []
        tools_list += await self.env.get_tools()  # add env tools
        # TODO: handle duplicate tool names
        for toolkit_config in self.config.toolkits:
            if toolkit_config.mode == "mcp":
                await self._load_mcp_server(toolkit_config)
            elif toolkit_config.mode == "builtin":
                toolkit = await self._load_toolkit(toolkit_config)
                tools_list.extend(await toolkit.get_tools_in_agents())
            else:
                raise ValueError(f"Unknown toolkit mode: {toolkit_config.mode}")
        tool_names = [tool.name for tool in tools_list]
        logger.info(f"Loaded {len(tool_names)} tools: {tool_names}")
        self.tools = tools_list
        return tools_list

    async def _load_toolkit(self, toolkit_config: ToolkitConfig) -> AsyncBaseToolkit:
        logger.info(f"Loading builtin toolkit `{toolkit_config.name}` with config {toolkit_config}")
        toolkit = await self._tools_exit_stack.enter_async_context(TOOLKIT_MAP[toolkit_config.name](toolkit_config))
        self._toolkits.append(toolkit)
        return toolkit

    async def _load_mcp_server(self, toolkit_config: ToolkitConfig) -> MCPServer:
        logger.info(f"Loading MCP server `{toolkit_config.name}` with params {toolkit_config.config}")
        server = await self._mcps_exit_stack.enter_async_context(
            MCPServerStdio(  # FIXME: support other types of servers
                name=toolkit_config.name,
                params=toolkit_config.config,
                client_session_timeout_seconds=20,
            )
        )
        self._mcp_servers.append(server)
        return server

    def _get_run_config(self) -> RunConfig:
        run_config = RunConfig(
            model=self.current_agent.model,
            model_settings=self.config.model.model_settings,
            workflow_name=self.config.agent.name,
        )
        return run_config

    def _get_context(self) -> dict:
        return {
            "context_manager": self.context_manager,
            "env": self.env,
        }

    # wrap `Runner` apis in @openai-agents
    async def run(self, input: str | list[TResponseInputItem], trace_id: str = None) -> TaskRecorder:
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        task_recorder = TaskRecorder(input, trace_id)
        run_kwargs = {
            "starting_agent": self.current_agent,
            "input": input,
            "context": self._get_context(),
            "max_turns": self.config.max_turns,
            "hooks": self._run_hooks,
            "run_config": self._get_run_config(),
        }
        if AgentsUtils.get_current_trace():
            run_result = await Runner.run(**run_kwargs)
        else:
            with trace(workflow_name="simple_agent", trace_id=trace_id):
                run_result = await Runner.run(**run_kwargs)
        task_recorder.add_run_result(run_result)
        task_recorder.set_final_output(run_result.final_output)
        return task_recorder

    def run_streamed(self, input: str | list[TResponseInputItem], trace_id: str = None) -> RunResultStreaming:
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        run_kwargs = {
            "starting_agent": self.current_agent,
            "input": input,
            "context": self._get_context(),
            "max_turns": self.config.max_turns,
            "hooks": self._run_hooks,
            "run_config": self._get_run_config(),
        }
        if AgentsUtils.get_current_trace():
            return Runner.run_streamed(**run_kwargs)
        else:
            with trace(workflow_name="simple_agent", trace_id=trace_id):
                return Runner.run_streamed(**run_kwargs)

    # util apis
    async def chat(self, input: str) -> RunResult:
        # TODO: set "session-level" tracing for multi-turn chat
        self.input_items.append({"content": input, "role": "user"})
        # print(f"< {self.input_items}")
        recorder = await self.run(self.input_items)
        run_result = recorder.get_run_result()
        AgentsUtils.print_new_items(run_result.new_items)
        self.input_items = run_result.to_input_list()
        self.current_agent = run_result.last_agent
        return run_result

    async def chat_streamed(self, input: str) -> RunResultStreaming:
        self.input_items.append({"content": input, "role": "user"})
        run_result_streaming = self.run_streamed(self.input_items)
        await AgentsUtils.print_stream_events(run_result_streaming.stream_events())
        self.input_items = run_result_streaming.to_input_list()
        self.current_agent = run_result_streaming.last_agent
        return run_result_streaming

    def set_run_hooks(self, run_hooks: RunHooks):
        # WIP
        self._run_hooks = run_hooks
