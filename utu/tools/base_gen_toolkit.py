import asyncio

from ..config import ToolkitConfig
from ..utils import SimplifiedAsyncOpenAI, get_logger, oneline_object
from .base import TOOL_PROMPTS, AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class BaseGeneratorToolkit(AsyncBaseToolkit):
    """Base Generator Toolkit

    NOTE:
        - Please configure the required env variables! See `configs/agents/tools/search.yaml`

    Methods:
        - direct_reply(query: str)
    """

    def __init__(self, config: ToolkitConfig = None):
        super().__init__(config)
        # llm for qa
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )

    @register_tool
    async def direct_reply(self, query: str) -> dict:
        """A direct-reply engine that accepts text input and produces an immediate, step-by-step answer.
        - Use it for general queries or tasks that do NOT require specialized knowledge or specific tools in the
toolbox.
        - This tool requires clear, specific query.
        - Use it to answer the original query through step by step reasoning for tasks without complex or
multi-step reasoning.
        - For complex queries, break them down into subtasks and use the tool multiple times.
        - Use it only as a starting point for complex tasks, then refine with specialized tools.
        - Verify or double-check important information from its responses as this tool may provide hallucinated or incorrect responses.

        Args:
            query (str): User question or prompt.
        """
        logger.info(f"[tool] direct reply: {oneline_object(query)}")
        res = await self._qa(query)
        logger.info(oneline_object(res))
        return res
        

    async def _qa(self, query: str) -> str:
        return await self.llm.query_one(
            messages=[{"role": "user", "content": query}], **self.config.config_llm.model_params.model_dump()
        )