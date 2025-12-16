from typing import Literal

# from openai import NOT_GIVEN
from agents import ModelSettings
from pydantic import ConfigDict, Field

from ..utils import EnvUtils
from .base_config import ConfigBaseModel


class ModelProviderConfig(ConfigBaseModel):
    """config for model provider"""

    type: Literal["chat.completions", "responses", "litellm"] = "chat.completions"
    """model type, supported types: chat.completions, responses"""
    model: str = EnvUtils.get_env("UTU_LLM_MODEL")
    """model name"""
    base_url: str | None = None
    """model provider base url"""
    api_key: str | None = None
    """model provider api key"""


class ModelSettingsConfig(ConfigBaseModel, ModelSettings):
    """ModelSettings in openai-agents"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelParamsConfig(ConfigBaseModel):
    """Basic params shared in chat.completions and responses"""

    temperature: float | None = None
    top_p: float | None = None
    parallel_tool_calls: bool | None = None


class ModelConfigs(ConfigBaseModel):
    """Overall model config"""

    model_provider: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    """config for model provider"""
    model_settings: ModelSettingsConfig = Field(default_factory=ModelSettingsConfig)
    """config for agent's model settings"""
    model_params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    """config for basic model usage, e.g. `query_one` in tools / judger"""

    termination_max_tokens: int | None = None
    """max tokens for the model, used in truncation logic"""
