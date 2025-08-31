import logging
import os
from typing import Literal

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseStreamEvent

from .types import (
    OpenAIChatCompletionParams,
    OpenAIChatCompletionParamsKeys,
    OpenAIResponsesParams,
    OpenAIResponsesParamsKeys,
)

logger = logging.getLogger(__name__)


class SimplifiedAsyncOpenAI(AsyncOpenAI):
    """Simplified OpenAI client for chat.completions and responses API, with default config"""

    def __init__(
        self,
        *,
        type: Literal["chat.completions", "responses"] = None,
        # openai client kwargs
        api_key: str | None = None,
        base_url: str | None = None,
        # default configs
        **kwargs: dict,
    ) -> None:
        logger.info(f"> type: {type}, base_url: {base_url}, kwargs: {kwargs}")
        super().__init__(
            api_key=api_key or os.getenv("UTU_LLM_API_KEY") or "xxx", base_url=base_url or os.getenv("UTU_LLM_BASE_URL")
        )
        self.type = type or os.getenv("UTU_LLM_TYPE", "chat.completions")
        self.type_create_params = (
            OpenAIChatCompletionParamsKeys if self.type == "chat.completions" else OpenAIResponsesParamsKeys
        )
        self.default_config = self._process_kwargs(kwargs)

    def _process_kwargs(self, kwargs: dict) -> dict:
        # parse kwargs for ChatCompletionParams
        default_config = {}
        for k, v in kwargs.items():
            if k in self.type_create_params:
                default_config[k] = v
        if "model" not in default_config:
            default_config["model"] = os.getenv("UTU_LLM_MODEL")
        return default_config

    async def query_one(self, **kwargs) -> str:
        """Simplified chat.complete / responses API
        WARNING: Only for basic text i/o usage! You should not use the method with querying with customized configs!
        """
        if "stream" in kwargs:
            assert kwargs["stream"] is False, "stream is not supported in `query_one`"

        if self.type == "chat.completions":
            chat_completion: ChatCompletion = await self.chat_completions_create(**kwargs)
            return chat_completion.choices[0].message.content
        elif self.type == "responses":
            response: Response = await self.responses_create(**kwargs)
            return response.output_text  # NOTE: will not return toolcall or reasoning
        else:
            raise ValueError(f"Unknown type: {self.type}")

    async def chat_completions_create(self, **kwargs) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        assert self.type == "chat.completions", "`chat_completions_create` is not supported for responses API"
        unknown_params = self.check_known_keys(kwargs, self.type_create_params)
        if unknown_params:
            logger.warning(f"Unknown parameters: {unknown_params} for {self.type} API!")
        kwargs = self.process_chat_completion_params(kwargs, self.default_config)
        return await self.chat.completions.create(**kwargs)

    async def responses_create(self, **kwargs) -> Response | AsyncStream[ResponseStreamEvent]:
        unknown_params = self.check_known_keys(kwargs, self.type_create_params)
        if unknown_params - {"messages"}:  # ignore
            logger.warning(f"Unknown parameters: {unknown_params} for {self.type} API!")
        assert self.type == "responses", "`responses_create` is not supported for chat.completions API"
        kwargs = self.process_responses_params(kwargs, self.default_config)
        return await self.responses.create(**kwargs)

    def process_chat_completion_params(
        self, kwargs: OpenAIChatCompletionParams, default_config: OpenAIChatCompletionParams
    ) -> OpenAIChatCompletionParams:
        """Process chat completion params, convert str to list of messages, merge default config"""
        assert "messages" in kwargs
        if isinstance(kwargs["messages"], str):
            kwargs["messages"] = [{"role": "user", "content": kwargs["messages"]}]
        return self._merge_default_config(kwargs, default_config)

    def process_responses_params(
        self, kwargs: OpenAIResponsesParams, default_config: OpenAIResponsesParams
    ) -> OpenAIResponsesParams:
        """Process responses params, convert str to list of messages, merge default config"""
        if "input" not in kwargs:
            # try parse query for chat.completions
            assert "messages" in kwargs
            input = kwargs.pop("messages")
            if isinstance(input, str):
                kwargs["input"] = [{"role": "user", "content": input}]
            else:
                kwargs["input"] = input
        else:
            if isinstance(kwargs["input"], str):
                kwargs["input"] = [{"role": "user", "content": kwargs["input"]}]
        return self._merge_default_config(kwargs, default_config)

    def _merge_default_config(self, kwargs: dict, default_config: dict) -> dict:
        """Merge default config"""
        for k, v in default_config.items():
            if k not in kwargs:
                kwargs[k] = v
        return kwargs

    def check_known_keys(self, kwargs: dict, known_keys: set[str]) -> set:
        """Check if all keys in kwargs are in known_keys"""
        unknown_keys = set(kwargs.keys()) - known_keys
        return unknown_keys
