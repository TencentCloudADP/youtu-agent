"""
ADD:
- inject context infos in `_run_single_turn` and `_run_single_turn_streamed`
- add "context_manager" logic for `_run_single_turn` and `_run_single_turn_streamed`
- add termination logic based on `termination_max_tokens` in `ModelConfigs`
"""

import asyncio
import logging
from typing import cast
from typing_extensions import Unpack

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
from agents._run_impl import (
    RunImpl, NextStepHandoff, TraceCtxManager, NextStepFinalOutput, NextStepRunAgain, 
    get_model_tracing_impl, 
)
from agents.exceptions import ModelBehaviorError, MaxTurnsExceeded, AgentsException, RunErrorDetails
from agents.guardrail import InputGuardrailResult
from agents.items import HandoffCallItem, ModelResponse, ToolCallItem, ToolCallItemTypes, ReasoningItem
from agents.result import RunResult
from agents.run import (
    AgentRunner, AgentToolUseTracker, RunResultStreaming, SingleStepResult, RunOptions,
    _TOOL_CALL_TYPES, _ServerConversationTracker, _copy_str_or_list, DEFAULT_MAX_TURNS
)
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from agents.tool_guardrails import ToolInputGuardrailResult, ToolOutputGuardrailResult
from agents.tracing import Span, AgentSpanData, SpanError, agent_span
from agents.util import _coro, _error_tracing
from agents.usage import Usage
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
)

from ..config import AgentConfig
from ..context import BaseContextManager

logger = logging.getLogger(__name__)


class UTUAgentRunner(AgentRunner):

    async def run(
        self,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        **kwargs: Unpack[RunOptions[TContext]],
    ) -> RunResult:
        context = kwargs.get("context")
        max_turns = kwargs.get("max_turns", DEFAULT_MAX_TURNS)
        hooks = cast(RunHooks[TContext], self._validate_run_hooks(kwargs.get("hooks")))
        run_config = kwargs.get("run_config")
        previous_response_id = kwargs.get("previous_response_id")
        conversation_id = kwargs.get("conversation_id")
        session = kwargs.get("session")
        if run_config is None:
            run_config = RunConfig()

        if conversation_id is not None or previous_response_id is not None:
            server_conversation_tracker = _ServerConversationTracker(
                conversation_id=conversation_id, previous_response_id=previous_response_id
            )
        else:
            server_conversation_tracker = None

        # Keep original user input separate from session-prepared input
        original_user_input = input
        prepared_input = await self._prepare_input_with_session(
            input, session, run_config.session_input_callback
        )

        tool_use_tracker = AgentToolUseTracker()

        with TraceCtxManager(
            workflow_name=run_config.workflow_name,
            trace_id=run_config.trace_id,
            group_id=run_config.group_id,
            metadata=run_config.trace_metadata,
            disabled=run_config.tracing_disabled,
        ):
            current_turn = 0
            original_input: str | list[TResponseInputItem] = _copy_str_or_list(prepared_input)
            generated_items: list[RunItem] = []
            model_responses: list[ModelResponse] = []

            context_wrapper: RunContextWrapper[TContext] = RunContextWrapper(
                context=context,  # type: ignore
            )

            input_guardrail_results: list[InputGuardrailResult] = []
            tool_input_guardrail_results: list[ToolInputGuardrailResult] = []
            tool_output_guardrail_results: list[ToolOutputGuardrailResult] = []

            current_span: Span[AgentSpanData] | None = None
            current_agent = starting_agent
            should_run_agent_start_hooks = True

            # save only the new user input to the session, not the combined history
            await self._save_result_to_session(session, original_user_input, [])

            try:
                while True:
                    all_tools = await AgentRunner._get_all_tools(current_agent, context_wrapper)

                    # Start an agent span if we don't have one. This span is ended if the current
                    # agent changes, or if the agent loop ends.
                    if current_span is None:
                        handoff_names = [
                            h.agent_name
                            for h in await AgentRunner._get_handoffs(current_agent, context_wrapper)
                        ]
                        if output_schema := AgentRunner._get_output_schema(current_agent):
                            output_type_name = output_schema.name()
                        else:
                            output_type_name = "str"

                        current_span = agent_span(
                            name=current_agent.name,
                            handoffs=handoff_names,
                            output_type=output_type_name,
                        )
                        current_span.start(mark_as_current=True)
                        current_span.span_data.tools = [t.name for t in all_tools]

                    current_turn += 1
                    if current_turn > max_turns:
                        _error_tracing.attach_error_to_span(
                            current_span,
                            SpanError(
                                message="Max turns exceeded",
                                data={"max_turns": max_turns},
                            ),
                        )
                        raise MaxTurnsExceeded(f"Max turns ({max_turns}) exceeded")

                    logger.debug(
                        f"Running agent {current_agent.name} (turn {current_turn})",
                    )
                    # ADD: inject context infos
                    if isinstance(context_wrapper.context, dict):
                        context_wrapper.context.update({
                            "current_turn": current_turn,
                            "max_turns": max_turns,
                        })

                    if current_turn == 1:
                        input_guardrail_results, turn_result = await asyncio.gather(
                            self._run_input_guardrails(
                                starting_agent,
                                starting_agent.input_guardrails
                                + (run_config.input_guardrails or []),
                                _copy_str_or_list(prepared_input),
                                context_wrapper,
                            ),
                            self._run_single_turn(
                                agent=current_agent,
                                all_tools=all_tools,
                                original_input=original_input,
                                generated_items=generated_items,
                                hooks=hooks,
                                context_wrapper=context_wrapper,
                                run_config=run_config,
                                should_run_agent_start_hooks=should_run_agent_start_hooks,
                                tool_use_tracker=tool_use_tracker,
                                server_conversation_tracker=server_conversation_tracker,
                            ),
                        )
                    else:
                        turn_result = await self._run_single_turn(
                            agent=current_agent,
                            all_tools=all_tools,
                            original_input=original_input,
                            generated_items=generated_items,
                            hooks=hooks,
                            context_wrapper=context_wrapper,
                            run_config=run_config,
                            should_run_agent_start_hooks=should_run_agent_start_hooks,
                            tool_use_tracker=tool_use_tracker,
                            server_conversation_tracker=server_conversation_tracker,
                        )
                    should_run_agent_start_hooks = False

                    model_responses.append(turn_result.model_response)
                    original_input = turn_result.original_input
                    generated_items = turn_result.generated_items

                    if server_conversation_tracker is not None:
                        server_conversation_tracker.track_server_items(turn_result.model_response)

                    # Collect tool guardrail results from this turn
                    tool_input_guardrail_results.extend(turn_result.tool_input_guardrail_results)
                    tool_output_guardrail_results.extend(turn_result.tool_output_guardrail_results)

                    if isinstance(turn_result.next_step, NextStepFinalOutput):
                        output_guardrail_results = await self._run_output_guardrails(
                            current_agent.output_guardrails + (run_config.output_guardrails or []),
                            current_agent,
                            turn_result.next_step.output,
                            context_wrapper,
                        )
                        result = RunResult(
                            input=original_input,
                            new_items=generated_items,
                            raw_responses=model_responses,
                            final_output=turn_result.next_step.output,
                            _last_agent=current_agent,
                            input_guardrail_results=input_guardrail_results,
                            output_guardrail_results=output_guardrail_results,
                            tool_input_guardrail_results=tool_input_guardrail_results,
                            tool_output_guardrail_results=tool_output_guardrail_results,
                            context_wrapper=context_wrapper,
                        )
                        if not any(
                            guardrail_result.output.tripwire_triggered
                            for guardrail_result in input_guardrail_results
                        ):
                            await self._save_result_to_session(
                                session, [], turn_result.new_step_items
                            )

                        return result
                    elif isinstance(turn_result.next_step, NextStepHandoff):
                        current_agent = cast(Agent[TContext], turn_result.next_step.new_agent)
                        current_span.finish(reset_current=True)
                        current_span = None
                        should_run_agent_start_hooks = True
                    elif isinstance(turn_result.next_step, NextStepRunAgain):
                        if not any(
                            guardrail_result.output.tripwire_triggered
                            for guardrail_result in input_guardrail_results
                        ):
                            await self._save_result_to_session(
                                session, [], turn_result.new_step_items
                            )
                    else:
                        raise AgentsException(
                            f"Unknown next step type: {type(turn_result.next_step)}"
                        )
            except AgentsException as exc:
                exc.run_data = RunErrorDetails(
                    input=original_input,
                    new_items=generated_items,
                    raw_responses=model_responses,
                    last_agent=current_agent,
                    context_wrapper=context_wrapper,
                    input_guardrail_results=input_guardrail_results,
                    output_guardrail_results=[],
                )
                raise
            finally:
                if current_span:
                    current_span.finish(reset_current=True)

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
        server_conversation_tracker: _ServerConversationTracker | None = None,
    ) -> SingleStepResult:
        emitted_tool_call_ids: set[str] = set()
        emitted_reasoning_item_ids: set[str] = set()

        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_agent_start(context_wrapper, agent),
                (
                    agent.hooks.on_start(context_wrapper, agent)
                    if agent.hooks
                    else _coro.noop_coroutine()
                ),
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

        if server_conversation_tracker is not None:
            input = server_conversation_tracker.prepare_input(
                streamed_result.input, streamed_result.new_items
            )
        else:
            input = ItemHelpers.input_to_new_input_list(streamed_result.input)
            input.extend([item.to_input_item() for item in streamed_result.new_items])

        # ADD: inject context infos
        if isinstance(context_wrapper.context, dict):
            context_wrapper.context.update({
                "current_turn": streamed_result.current_turn,
                "max_turns": streamed_result.max_turns,
                # "streamed_result": streamed_result
            })
        input = cls._context_manager_preprocess(input, context_wrapper)

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
                agent.hooks.on_llm_start(
                    context_wrapper, agent, filtered.instructions, filtered.input
                )
                if agent.hooks
                else _coro.noop_coroutine()
            ),
        )

        previous_response_id = (
            server_conversation_tracker.previous_response_id
            if server_conversation_tracker
            else None
        )
        conversation_id = (
            server_conversation_tracker.conversation_id if server_conversation_tracker else None
        )

        # 1. Stream the output events
        async for event in model.stream_response(
            filtered.instructions,
            filtered.input,
            model_settings,
            all_tools,
            output_schema,
            handoffs,
            get_model_tracing_impl(
                run_config.tracing_disabled, run_config.trace_include_sensitive_data
            ),
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt_config,
        ):
            # Emit the raw event ASAP
            streamed_result._event_queue.put_nowait(RawResponsesStreamEvent(data=event))

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
                    call_id: str | None = getattr(
                        output_item, "call_id", getattr(output_item, "id", None)
                    )

                    if call_id and call_id not in emitted_tool_call_ids:
                        emitted_tool_call_ids.add(call_id)

                        tool_item = ToolCallItem(
                            raw_item=cast(ToolCallItemTypes, output_item),
                            agent=agent,
                        )
                        streamed_result._event_queue.put_nowait(
                            RunItemStreamEvent(item=tool_item, name="tool_called")
                        )

                elif isinstance(output_item, ResponseReasoningItem):
                    reasoning_id: str | None = getattr(output_item, "id", None)

                    if reasoning_id and reasoning_id not in emitted_reasoning_item_ids:
                        emitted_reasoning_item_ids.add(reasoning_id)

                        reasoning_item = ReasoningItem(raw_item=output_item, agent=agent)
                        streamed_result._event_queue.put_nowait(
                            RunItemStreamEvent(item=reasoning_item, name="reasoning_item_created")
                        )

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

        # ADD: terminate when context too long
        final_response = cls._check_too_long(final_response, context_wrapper)

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
                    and (
                        call_id := getattr(
                            item.raw_item, "call_id", getattr(item.raw_item, "id", None)
                        )
                    )
                    and call_id in emitted_tool_call_ids
                )
            ]

        if emitted_reasoning_item_ids:
            # Filter out reasoning items that were already emitted during streaming
            items_to_filter = [
                item
                for item in items_to_filter
                if not (
                    isinstance(item, ReasoningItem)
                    and (reasoning_id := getattr(item.raw_item, "id", None))
                    and reasoning_id in emitted_reasoning_item_ids
                )
            ]

        # Filter out HandoffCallItem to avoid duplicates (already sent earlier)
        items_to_filter = [
            item for item in items_to_filter if not isinstance(item, HandoffCallItem)
        ]

        # Create filtered result and send to queue
        filtered_result = _dc.replace(single_step_result, new_step_items=items_to_filter)
        RunImpl.stream_step_result_to_queue(filtered_result, streamed_result._event_queue)
        return single_step_result

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
        server_conversation_tracker: _ServerConversationTracker | None = None,
    ) -> SingleStepResult:
        # Ensure we run the hooks before anything else
        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_agent_start(context_wrapper, agent),
                (
                    agent.hooks.on_start(context_wrapper, agent)
                    if agent.hooks
                    else _coro.noop_coroutine()
                ),
            )

        system_prompt, prompt_config = await asyncio.gather(
            agent.get_system_prompt(context_wrapper),
            agent.get_prompt(context_wrapper),
        )

        output_schema = cls._get_output_schema(agent)
        handoffs = await cls._get_handoffs(agent, context_wrapper)
        if server_conversation_tracker is not None:
            input = server_conversation_tracker.prepare_input(original_input, generated_items)
        else:
            input = ItemHelpers.input_to_new_input_list(original_input)
            input.extend([generated_item.to_input_item() for generated_item in generated_items])

        # ADD: context manager
        input = cls._context_manager_preprocess(input, context_wrapper)

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
            server_conversation_tracker,
            prompt_config,
        )

        # ADD: terminate when context too long
        new_response = cls._check_too_long(new_response, context_wrapper)

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
    def _context_manager_preprocess(
        cls, input: list[TResponseInputItem], context_wrapper: RunContextWrapper[TContext]
    ) -> list[TResponseInputItem]:
        if context_wrapper.context:
            context_manager: BaseContextManager = context_wrapper.context.get("context_manager", None)
            input = context_manager.preprocess(input, context_wrapper)
        return input
        # print(f"< [DEBUG] input: {input}")

    @classmethod
    def _check_too_long(
        cls, new_response: ModelResponse, context_wrapper: RunContextWrapper[TContext]
    ) -> ModelResponse:
        # ADD: terminate when response too long. before `_get_single_step_result_from_response`
        """Check if the context is too long. If the total_tokens exceed the max limit, terminate this rollout
        (by removing all tool calls from the response)"""
        if context_wrapper.context is None:
            return new_response
        config: AgentConfig = context_wrapper.context.get("agent_config", None)
        if not config or not config.model.termination_max_tokens:
            return new_response

        MAX_TOKENS = config.model.termination_max_tokens
        total_tokens = new_response.usage.total_tokens if new_response.usage else 0
        logger.warning(f"> [DEBUG] total_tokens: {total_tokens}, MAX_TOKENS: {MAX_TOKENS}")
        if total_tokens > MAX_TOKENS:
            output = []
            for item in new_response.output:
                if isinstance(item, ResponseOutputMessage):
                    output.append(item)
                elif isinstance(item, ResponseFunctionToolCall):
                    pass
            logger.warning(
                f"Response truncated due to exceeding max token limit of {MAX_TOKENS}.\n"
                f"  Raw response: {new_response.output}."
            )
            new_response.output = output
        return new_response
