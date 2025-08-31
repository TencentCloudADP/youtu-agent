import asyncio
import logging

from agents import (
    Agent,
    ItemHelpers,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    RunItem,
    TContext,
    Tool,
    TResponseInputItem,
)
from agents.run import AgentRunner, AgentToolUseTracker, SingleStepResult
from agents.util import _coro

from ..context import BaseContextManager

logger = logging.getLogger(__name__)


class UTUAgentRunner(AgentRunner):
    # TODO: also add context_manager to _run_single_turn_streamed for .run_streamed
    @classmethod
    async def _run_single_turn(
        cls,
        *,
        agent: Agent[TContext],
        all_tools: list[Tool],
        original_input: str | list[TResponseInputItem],
        generated_items: list[RunItem],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_hooks: bool,
        tool_use_tracker: AgentToolUseTracker,
        previous_response_id: str | None,
    ) -> SingleStepResult:
        # Ensure we run the hooks before anything else
        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_agent_start(context_wrapper, agent),
                (agent.hooks.on_start(context_wrapper, agent) if agent.hooks else _coro.noop_coroutine()),
            )

        system_prompt, prompt_config = await asyncio.gather(
            agent.get_system_prompt(context_wrapper),
            agent.get_prompt(context_wrapper),
        )

        output_schema = cls._get_output_schema(agent)
        handoffs = await cls._get_handoffs(agent, context_wrapper)
        input = ItemHelpers.input_to_new_input_list(original_input)
        input.extend([generated_item.to_input_item() for generated_item in generated_items])

        # FIXME: set context manage as a hook?
        # ADD: context manager
        context_manager: BaseContextManager | None = context_wrapper.context.get("context_manager", None)
        if context_manager:
            input = await _coro.run_in_executor(context_manager.preprocess, input, context_wrapper)
        # print(f"< [DEBUG] input: {input}")
        new_response = await cls._get_new_response(
            agent,
            system_prompt,
            input,
            output_schema,
            all_tools,
            handoffs,
            context_wrapper,
            run_config,
            tool_use_tracker,
            previous_response_id,
            prompt_config,
        )

        # ADD: response logging
        # print(json.dumps([item.model_dump() for item in new_response.output], ensure_ascii=False))

        single_turn_result = await cls._get_single_step_result_from_response(
            agent=agent,
            original_input=original_input,
            pre_step_items=generated_items,
            new_response=new_response,
            output_schema=output_schema,
            all_tools=all_tools,
            handoffs=handoffs,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
            tool_use_tracker=tool_use_tracker,
        )
        # ADD: postprocess
        if context_manager:
            single_turn_result = await _coro.run_in_executor(context_manager.postprocess, single_turn_result)
        return single_turn_result
