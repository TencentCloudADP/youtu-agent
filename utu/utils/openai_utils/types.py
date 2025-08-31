from collections.abc import Iterable
from typing import Literal, TypedDict

import httpx
from openai._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.responses import ResponseInputParam, ResponseTextConfigParam, ToolParam
from openai.types.responses.response_create_params import ToolChoice
from openai.types.shared import ChatModel, Reasoning, ReasoningEffort, ResponsesModel


class OpenAICreateBaseParams(TypedDict):
    stream: bool | None = False
    # from openai.resources.chat.completions.Completions.create
    extra_headers: Headers | None = None
    extra_query: Query | None = None
    extra_body: Body | None = None
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN


# CompletionCreateParams | https://platform.openai.com/docs/api-reference/chat/create
class OpenAIChatCompletionParams(TypedDict, OpenAICreateBaseParams):
    # NOTE: only for typing
    messages: Iterable[ChatCompletionMessageParam]  # required
    model: str | ChatModel  # required
    frequency_penalty: float | None
    logit_bias: dict[str, int] | None
    logprobs: bool | None
    max_completion_tokens: int | None
    max_tokens: int | None
    n: int | None
    presence_penalty: float | None
    reasoning_effort: ReasoningEffort | None
    response_format: ResponseFormat
    seed: int | None
    temperature: float | None
    top_p: float | None
    tools: list[ChatCompletionToolParam] | None
    tool_choice: Literal["none", "auto", "required"] | None
    parallel_tool_calls: bool | None
    stop: str | list[str] | None
    top_logprobs: int | None


# ResponseCreateParams | https://platform.openai.com/docs/api-reference/responses/create
class OpenAIResponsesParams(TypedDict, OpenAICreateBaseParams):
    input: str | ResponseInputParam
    instructions: str | None
    max_output_tokens: int | None
    max_tool_calls: int | None
    model: ResponsesModel
    parallel_tool_calls: bool | None
    previous_response_id: str | None
    reasoning: Reasoning | None
    temperature: float | None
    text: ResponseTextConfigParam
    tool_choice: ToolChoice
    tools: Iterable[ToolParam]
    top_logprobs: int | None
    top_p: float | None
    truncation: Literal["auto", "disabled"] | None


OpenAIChatCompletionParamsKeys = OpenAIChatCompletionParams.__annotations__.keys()
OpenAIResponsesParamsKeys = OpenAIResponsesParams.__annotations__.keys()
