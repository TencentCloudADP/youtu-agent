import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Literal

from agents import (
    FunctionTool,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    Model,
    ModelSettings,
    ModelTracing,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    ReasoningItem,
    RunItem,
    RunResult,
    StreamEvent,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
)
from agents.models.chatcmpl_converter import Converter
from agents.stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent, RunItemStreamEvent
from agents.tracing import Trace, gen_trace_id, get_current_trace
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.responses import ResponseFunctionToolCall

from .image import encode_image
from .openai_utils import OpenAIChatCompletionParams
from .print_utils import PrintUtils

logger = logging.getLogger(__name__)


class ChatCompletionConverter(Converter):
    @classmethod
    def items_to_messages(cls, items: str | Iterable[TResponseInputItem]) -> list[ChatCompletionMessageParam]:
        # skip reasoning, see chatcmpl_converter.Converter.items_to_messages()
        # agents.exceptions.UserError: Unhandled item type or structure:
        # {'id': '__fake_id__', 'summary': [{'text': '...', 'type': 'summary_text'}], 'type': 'reasoning'}
        if not isinstance(items, str):  # TODO: check it!
            items = cls.filter_items(items)
        return Converter.items_to_messages(items)

    @classmethod
    def filter_items(cls, items: str | Iterable[TResponseInputItem]) -> str | list[TResponseInputItem]:
        if isinstance(items, str):
            return items
        filtered_items = []
        for item in items:
            if item.get("type", None) == "reasoning":
                continue
            filtered_items.append(item)
        return filtered_items

    @classmethod
    def items_to_dict(cls, items: str | Iterable[TResponseInputItem]) -> list[dict]:
        """convert items to a list of dict which have {"role", "content"}
        WIP!
        """
        if isinstance(items, str):
            return [{"role": "user", "content": items}]
        result = []
        for item in items:
            if msg := Converter.maybe_easy_input_message(item):
                result.append(msg)
            elif msg := Converter.maybe_input_message(item):
                result.append(msg)
            elif msg := Converter.maybe_response_output_message(item):
                result.append(msg)
            elif msg := Converter.maybe_file_search_call(item):
                msg.update({"role": "tool", "content": msg["results"]})
                result.append(msg)
            elif msg := Converter.maybe_function_tool_call(item):
                msg.update({"role": "assistant", "content": f"{msg['name']}({msg['arguments']})"})
                result.append(msg)
            elif msg := Converter.maybe_function_tool_call_output(item):
                msg.update({"role": "tool", "content": msg["output"], "tool_call_id": msg["call_id"]})
                result.append(msg)
            elif msg := Converter.maybe_reasoning_message(item):
                msg.update({"role": "assistant", "content": msg["summary"]})
                result.append(msg)
            else:
                logger.warning(f"Unknown message type: {item}")
                result.append({"role": "assistant", "content": f"Unknown message type: {item}"})
        return result


class AgentsUtils:
    """Utils for openai-agents SDK"""

    @staticmethod
    def generate_group_id() -> str:
        """Generate a unique group ID. (Used in OpenAI tracing)
        Ref: https://openai.github.io/openai-agents-python/tracing/
        """
        return uuid.uuid4().hex[:16]

    @staticmethod
    def gen_trace_id() -> str:
        return gen_trace_id()

    @staticmethod
    def get_current_trace() -> Trace:
        return get_current_trace()

    @staticmethod
    def get_agents_model(
        type: Literal["responses", "chat.completions", "litellm"] = None,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
    ) -> Model:
        type = type or os.getenv("UTU_LLM_TYPE", "chat.completions")
        model = model or os.getenv("UTU_LLM_MODEL")
        if type == "litellm":
            # Ref: https://docs.litellm.ai/docs/providers
            # NOTE: should set .evn properly! e.g. AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION for Azure
            #   https://docs.litellm.ai/docs/providers/azure/
            from agents.extensions.models.litellm_model import LitellmModel

            return LitellmModel(model=model)

        base_url = base_url or os.getenv("UTU_LLM_BASE_URL")
        api_key = api_key or os.getenv("UTU_LLM_API_KEY")
        if not api_key or not base_url:
            raise ValueError("UTU_LLM_API_KEY and UTU_LLM_BASE_URL must be set")
        openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=100,
        )
        if type == "chat.completions":
            return OpenAIChatCompletionsModel(model=model, openai_client=openai_client)
        elif type == "responses":
            return OpenAIResponsesModel(model=model, openai_client=openai_client)
        else:
            raise ValueError("Invalid type: " + type)

    @staticmethod
    def get_trajectory_from_agent_result(agent_result: RunResult, agent_name: str = None) -> dict:
        if agent_name is None:
            agent_name = agent_result.last_agent.name
        return {
            "agent": agent_name,
            "trajectory": ChatCompletionConverter.items_to_messages(agent_result.to_input_list()),
        }

    @staticmethod
    def print_new_items(new_items: list[RunItem]) -> None:
        """Print new items generated by Runner.run()"""
        for new_item in new_items:
            agent_name = new_item.agent.name
            if isinstance(new_item, MessageOutputItem):
                PrintUtils.print_bot(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
            elif isinstance(new_item, HandoffOutputItem):
                PrintUtils.print_info(f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}")
            elif isinstance(new_item, ToolCallItem):
                assert isinstance(new_item.raw_item, ResponseFunctionToolCall)  # DONOT use openai's built-in tools
                PrintUtils.print_info(
                    f"{agent_name}: Calling a tool: {new_item.raw_item.name}({json.loads(new_item.raw_item.arguments)})"
                )
            elif isinstance(new_item, ToolCallOutputItem):
                PrintUtils.print_tool(f"Tool call output: {new_item.output}")
            elif isinstance(new_item, ReasoningItem):
                PrintUtils.print_info(f"{agent_name}: Reasoning: {new_item.raw_item}")
            else:
                PrintUtils.print_info(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")

    @staticmethod
    async def print_stream_events(result: AsyncIterator[StreamEvent]) -> None:
        """Print stream events generated by Runner.run_streamed()"""
        async for event in result:
            # print(f"> [DEBUG] event: {event}")
            if isinstance(event, RawResponsesStreamEvent):
                # event.data -- ResponseStreamEvent
                if event.data.type == "response.output_item.added":
                    match event.data.item.type:
                        # computer_call, code_interpreter_call, custom_tool_call, file_search_call, function_call,
                        # we_search_call, image_generation_call, local_shell_call,
                        # mcp_call, mcp_list_tools, mcp_approval_request, message, reasoning
                        case "message":
                            pass
                        case "function_call":
                            PrintUtils.print_bot(
                                f"<toolcall name={event.data.item.name}>{event.data.item.arguments}", end=""
                            )
                        case _:
                            PrintUtils.print_bot(f"<{event.data.item.type}>", end="")
                elif event.data.type == "response.output_item.done":
                    match event.data.item.type:
                        case "message":
                            pass
                            # PrintUtils.print_bot("")  # add a new line?
                        case "function_call":
                            PrintUtils.print_bot("</toolcall>")
                        case _:
                            # PrintUtils.print_bot(f"</{event.data.item.type}>")
                            logger.info(f"</{event.data.item.type}>")  # It seems that vllm's output order is wrong
                elif event.data.type == "response.content_part.added":
                    match event.data.part.type:
                        # output_text, refusal
                        case "output_text":
                            pass
                        case "refusal":
                            PrintUtils.print_bot(f"<refusal>{event.data.part.refusal}", end="")
                        case _:
                            logger.warning(f"Unknown part type: {event.data.part.type}! {event}")
                elif event.data.type == "response.content_part.done":
                    match event.data.part.type:
                        case "output_text":
                            pass
                        case "refusal":
                            PrintUtils.print_bot("</refusal>")
                        case _:
                            logger.warning(f"Unknown part type: {event.data.part.type}! {event}")
                elif event.data.type == "response.reasoning_summary_part.added":
                    PrintUtils.print_info("<reasoning_summary>", end="")
                elif event.data.type == "response.reasoning_summary_part.done":
                    # PrintUtils.print_info("</reasoning_summary>", end="")
                    logger.info("</reasoning_summary>")  # It seems that vllm's output order is wrong
                elif event.data.type == "response.reasoning_summary_text.delta":
                    PrintUtils.print_info(f"{event.data.delta}", end="")
                elif event.data.type == "response.function_call_arguments.delta":
                    PrintUtils.print_bot(f"{event.data.delta}", end="")
                elif event.data.type == "response.function_call_arguments.done":
                    pass
                elif event.data.type == "response.output_text.delta":
                    PrintUtils.print_bot(f"{event.data.delta}", end="")
                elif event.data.type == "response.reasoning_text.delta":
                    PrintUtils.print_info(f"{event.data.delta}", end="")
                elif event.data.type == "response.reasoning_text.done":
                    PrintUtils.print_info("</reasoning_text>", end="")
                elif event.data.type in ("response.output_text.done",):
                    PrintUtils.print_info("")
                elif event.data.type in (
                    "response.created",
                    "response.completed",
                    "response.in_progress",
                ):
                    pass
                else:
                    PrintUtils.print_info(f"Unknown event type: {event.data.type}! {event}")
                    # raise ValueError(f"Unknown event type: {event.data.type}")
            elif isinstance(event, RunItemStreamEvent):
                item: RunItem = event.item
                if item.type == "message_output_item":
                    pass  # do not print twice to avoid duplicate! (already handled `response.output_text.delta`)
                    # PrintUtils.print_bot(f"{ItemHelpers.text_message_output(item).strip()}")
                elif item.type == "reasoning_item":
                    pass
                elif item.type == "tool_call_item":
                    pass
                    # PrintUtils.print_bot([tool_call] {item.raw_item.name}({item.raw_item.arguments})")
                elif item.type == "tool_call_output_item":
                    PrintUtils.print_tool(f"[tool_output] {item.output}")  # item.raw_item
                elif item.type == "handoff_call_item":  # same as `ToolCallItem`
                    PrintUtils.print_bot(f"[handoff_call] {item.raw_item.name}({item.raw_item.arguments})")
                elif item.type == "handoff_output_item":
                    PrintUtils.print_info(f">> Handoff from {item.source_agent.name} to {item.target_agent.name}")
                elif event.type in (
                    "mcp_list_tools_item",
                    "mcp_approval_request_item",
                    "mcp_approval_response_item",
                ):
                    PrintUtils.print_info(f"  >>> Skipping item: {event}")
                else:
                    PrintUtils.print_info(f"  >>> Skipping item: {item.__class__.__name__}")
            elif isinstance(event, AgentUpdatedStreamEvent):
                PrintUtils.print_info(f">> new agent: {event.new_agent.name}")
            # skip events from youtu-agent
            elif event.type in ("orchestrator_stream_event", "orchestra_stream_event"):
                pass
            else:
                logger.warning(f"Unknown event type: {event.type}! {event}")
        print()  # Newline after stream?

    @staticmethod
    def convert_model_settings(params: OpenAIChatCompletionParams) -> ModelSettings:
        # "tools", "messages", "model"
        # FIXME: move to extra_args
        for p in ("max_completion_tokens", "top_logprobs", "logprobs", "seed", "stop"):
            if p in params:
                logger.warning(f"Parameter `{p}` is not supported in ModelSettings")
        return ModelSettings(
            max_tokens=params.get("max_tokens", None),
            temperature=params.get("temperature", None),
            top_p=params.get("top_p", None),
            frequency_penalty=params.get("frequency_penalty", None),
            presence_penalty=params.get("presence_penalty", None),
            tool_choice=params.get("tool_choice", None),
            parallel_tool_calls=params.get("parallel_tool_calls", None),
            extra_query=params.get("extra_query", None),
            extra_body=params.get("extra_body", None),
            extra_headers=params.get("extra_headers", None),
        )

    @staticmethod
    def convert_sp_input(
        messages: list[ChatCompletionMessageParam],
    ) -> tuple[str | None, str | list[TResponseInputItem]]:
        if isinstance(messages, str):
            return None, messages
        if messages[0].get("role", None) == "system":
            return messages[0]["content"], messages[1:]
        return None, messages

    @staticmethod
    def convert_tool(tool: ChatCompletionToolParam) -> FunctionTool:
        assert tool["type"] == "function"
        return FunctionTool(
            name=tool["function"]["name"],
            description=tool["function"].get("description", ""),
            params_json_schema=tool["function"].get("parameters", None),
            on_invoke_tool=None,
        )

    @staticmethod
    def get_message_from_image(image_url: str) -> dict:
        """Get a message dict for image input."""
        # from openai.types.responses.response_input_item_param import Message
        # from openai.types.responses.response_input_image_param import ResponseInputImageParam
        return {"role": "user", "content": [{"type": "input_image", "image_url": encode_image(image_url)}]}


class SimplifiedOpenAIChatCompletionsModel(OpenAIChatCompletionsModel):
    """extend OpenAIChatCompletionsModel to support basic api
    - enable tracing based on SimplifiedAsyncOpenAI
    """

    async def query_one(self, **kwargs) -> str:
        system_instructions, input = AgentsUtils.convert_sp_input(kwargs["messages"])
        model_settings = AgentsUtils.convert_model_settings(kwargs)
        tools = [AgentsUtils.convert_tool(tool) for tool in kwargs.get("tools", [])]
        response = await self.get_response(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.ENABLED,
            previous_response_id=None,
            prompt=None,
        )
        return ChatCompletionConverter.items_to_messages(response.to_input_items())
        # with generation_span(
        #     model=kwargs["model"],
        #     model_config=_model_settings,
        #     input=_messages,
        # ) as span_generation:
        #     result = await self.chat.completions.create(**kwargs)
        #     span_generation.span_data.output = result.choices[0].message.model_dump()
        #     return result
