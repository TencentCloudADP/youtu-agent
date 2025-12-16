from typing import Any

from agents import AgentBase, RunContextWrapper, RunHooks, TContext, Tool
from agents.tool_context import ToolContext
from typing_extensions import TypeVar

from ..utils import PrintUtils, get_logger

TAgent = TypeVar("TAgent", bound=AgentBase, default=AgentBase)
logger = get_logger(__name__)


class BaseRunHooks(RunHooks):
    def __init__(self):
        self.tool_result_max_length = 5000

    # on_llm_start, on_llm_end

    async def on_agent_start(self, context: RunContextWrapper[TContext], agent: TAgent) -> None:
        """Called before the agent is invoked. Called each time the current agent changes."""
        pass

    async def on_agent_end(self, context: RunContextWrapper[TContext], agent: TAgent, output: Any) -> None:
        """Called when the agent produces a final output."""
        pass

    async def on_handoff(self, context: RunContextWrapper[TContext], from_agent: TAgent, to_agent: TAgent) -> None:
        """Called when a handoff occurs."""
        pass

    async def on_tool_start(self, context: ToolContext, agent: TAgent, tool: Tool) -> None:
        """Called concurrently with tool invocation."""
        logger.debug(f"[toolcall-{context.tool_call_id}] {tool.name}({context.tool_arguments})")

    async def on_tool_end(self, context: ToolContext, agent: TAgent, tool: Tool, result: str) -> None:
        """Called after a tool is invoked."""
        logger.debug(f"[toolcall-{context.tool_call_id}] {tool.name}: {PrintUtils.truncate_text(result)}...")
        if len(result) > self.tool_result_max_length:
            logger.warning(f"Tool result too long! {len(result)} chars exceeds {self.tool_result_max_length}!")
