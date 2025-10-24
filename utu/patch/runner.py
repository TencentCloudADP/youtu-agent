import asyncio
import logging
from typing import cast

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
from agents._run_impl import RunImpl, get_model_tracing_impl
from agents.exceptions import ModelBehaviorError
from agents.items import HandoffCallItem, ModelResponse, ToolCallItem, ToolCallItemTypes
from agents.run import _TOOL_CALL_TYPES, AgentRunner, AgentToolUseTracker, RunResultStreaming, SingleStepResult
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from agents.usage import Usage
from agents.util import _coro
from openai.types.responses import ResponseCompletedEvent, ResponseOutputItemDoneEvent

from ..context import BaseContextManager

logger = logging.getLogger(__name__)


class UTUAgentRunner(AgentRunner):
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
        conversation_id: str | None,
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
        if context_wrapper.context:
            context_manager: BaseContextManager = context_wrapper.context.get("context_manager", None)
            input = context_manager.preprocess(input, context_wrapper)
        # print(f"< [DEBUG] input: {input}")

        new_response = await cls._get_new_response(
            agent,
            system_prompt,
            input,
            output_schema,
            all_tools,
            handoffs,
            hooks,
            context_wrapper,
            run_config,
            tool_use_tracker,
            previous_response_id,
            conversation_id,
            prompt_config,
        )

        return await cls._get_single_step_result_from_response(
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

    @classmethod
    async def _run_single_turn_streamed(
        cls,
        streamed_result: RunResultStreaming,
        agent: Agent[TContext],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_hooks: bool,
        tool_use_tracker: AgentToolUseTracker,
        all_tools: list[Tool],
        previous_response_id: str | None,
        conversation_id: str | None,
    ) -> SingleStepResult:
        emitted_tool_call_ids: set[str] = set()

        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_agent_start(context_wrapper, agent),
                (agent.hooks.on_start(context_wrapper, agent) if agent.hooks else _coro.noop_coroutine()),
            )

        output_schema = cls._get_output_schema(agent)

        streamed_result.current_agent = agent
        streamed_result._current_agent_output_schema = output_schema

        system_prompt, prompt_config = await asyncio.gather(
            agent.get_system_prompt(context_wrapper),
            agent.get_prompt(context_wrapper),
        )

        handoffs = await cls._get_handoffs(agent, context_wrapper)
        model = cls._get_model(agent, run_config)
        model_settings = agent.model_settings.resolve(run_config.model_settings)
        model_settings = RunImpl.maybe_reset_tool_choice(agent, tool_use_tracker, model_settings)

        final_response: ModelResponse | None = None

        input = ItemHelpers.input_to_new_input_list(streamed_result.input)
        input.extend([item.to_input_item() for item in streamed_result.new_items])

        # ADD: context manager
        if context_wrapper.context:
            context_manager: BaseContextManager = context_wrapper.context.get("context_manager", None)
            input = context_manager.preprocess(input, context_wrapper)

        # THIS IS THE RESOLVED CONFLICT BLOCK
        filtered = await cls._maybe_filter_model_input(
            agent=agent,
            run_config=run_config,
            context_wrapper=context_wrapper,
            input_items=input,
            system_instructions=system_prompt,
        )

        # Call hook just before the model is invoked, with the correct system_prompt.
        await asyncio.gather(
            hooks.on_llm_start(context_wrapper, agent, filtered.instructions, filtered.input),
            (
                agent.hooks.on_llm_start(context_wrapper, agent, filtered.instructions, filtered.input)
                if agent.hooks
                else _coro.noop_coroutine()
            ),
        )

        # 1. Stream the output events
        async for event in model.stream_response(
            filtered.instructions,
            filtered.input,
            model_settings,
            all_tools,
            output_schema,
            handoffs,
            get_model_tracing_impl(run_config.tracing_disabled, run_config.trace_include_sensitive_data),
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt_config,
        ):
            if isinstance(event, ResponseCompletedEvent):
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=event.response.usage.input_tokens,
                        output_tokens=event.response.usage.output_tokens,
                        total_tokens=event.response.usage.total_tokens,
                        input_tokens_details=event.response.usage.input_tokens_details,
                        output_tokens_details=event.response.usage.output_tokens_details,
                    )
                    if event.response.usage
                    else Usage()
                )
                final_response = ModelResponse(
                    output=event.response.output,
                    usage=usage,
                    response_id=event.response.id,
                )
                context_wrapper.usage.add(usage)

            if isinstance(event, ResponseOutputItemDoneEvent):
                output_item = event.item

                if isinstance(output_item, _TOOL_CALL_TYPES):
                    call_id: str | None = getattr(output_item, "call_id", getattr(output_item, "id", None))

                    if call_id and call_id not in emitted_tool_call_ids:
                        emitted_tool_call_ids.add(call_id)

                        tool_item = ToolCallItem(
                            raw_item=cast(ToolCallItemTypes, output_item),
                            agent=agent,
                        )
                        streamed_result._event_queue.put_nowait(RunItemStreamEvent(item=tool_item, name="tool_called"))

            streamed_result._event_queue.put_nowait(RawResponsesStreamEvent(data=event))

        # Call hook just after the model response is finalized.
        if final_response is not None:
            await asyncio.gather(
                (
                    agent.hooks.on_llm_end(context_wrapper, agent, final_response)
                    if agent.hooks
                    else _coro.noop_coroutine()
                ),
                hooks.on_llm_end(context_wrapper, agent, final_response),
            )

        # 2. At this point, the streaming is complete for this turn of the agent loop.
        if not final_response:
            raise ModelBehaviorError("Model did not produce a final response!")

        # 3. Now, we can process the turn as we do in the non-streaming case
        single_step_result = await cls._get_single_step_result_from_response(
            agent=agent,
            original_input=streamed_result.input,
            pre_step_items=streamed_result.new_items,
            new_response=final_response,
            output_schema=output_schema,
            all_tools=all_tools,
            handoffs=handoffs,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
            tool_use_tracker=tool_use_tracker,
            event_queue=streamed_result._event_queue,
        )

        import dataclasses as _dc

        # Filter out items that have already been sent to avoid duplicates
        items_to_filter = single_step_result.new_step_items

        if emitted_tool_call_ids:
            # Filter out tool call items that were already emitted during streaming
            items_to_filter = [
                item
                for item in items_to_filter
                if not (
                    isinstance(item, ToolCallItem)
                    and (call_id := getattr(item.raw_item, "call_id", getattr(item.raw_item, "id", None)))
                    and call_id in emitted_tool_call_ids
                )
            ]

        # Filter out HandoffCallItem to avoid duplicates (already sent earlier)
        items_to_filter = [item for item in items_to_filter if not isinstance(item, HandoffCallItem)]

        # Create filtered result and send to queue
        filtered_result = _dc.replace(single_step_result, new_step_items=items_to_filter)
        RunImpl.stream_step_result_to_queue(filtered_result, streamed_result._event_queue)
        return single_step_result
