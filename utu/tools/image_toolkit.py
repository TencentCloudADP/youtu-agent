"""
@smolagents/examples/open_deep_research/scripts/visual_qa.py
@camel/camel/toolkits/image_analysis_toolkit.py
https://platform.openai.com/docs/guides/images-vision?api-mode=chat
"""

import base64
from io import BytesIO
from urllib.parse import urlparse

import requests

try:
    from PIL import Image
except ImportError:
    pass

from ..config import ToolkitConfig
from ..utils import EnvUtils, SimplifiedAsyncOpenAI, get_logger
from .base import TOOL_PROMPTS, AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class ImageToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        image_llm_config = {
            "type": EnvUtils.get_env("UTU_IMAGE_LLM_TYPE", "chat.completions"),
            "model": EnvUtils.get_env("UTU_IMAGE_LLM_MODEL"),  # NOTE: you should set these envs in .env
            "api_key": EnvUtils.get_env("UTU_IMAGE_LLM_API_KEY"),
            "base_url": EnvUtils.get_env("UTU_IMAGE_LLM_BASE_URL"),
        }
        self.llm = SimplifiedAsyncOpenAI(**image_llm_config)

    def _load_image(self, image_path: str) -> str:
        parsed = urlparse(image_path)
        image: Image.Image = None

        if parsed.scheme in ("http", "https"):
            logger.debug(f"Fetching image from URL: {image_path}")
            try:
                response = requests.get(image_path, timeout=15)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except requests.exceptions.RequestException as e:
                logger.error(f"URL fetch failed: {e}")
                raise
        else:
            logger.debug(f"Loading local image: {image_path}")
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f"Image loading failed: {e}")
                raise ValueError(f"Invalid image file: {image_path}") from e
        # Convert the image to a base64 string
        buffer = BytesIO()
        image.save(buffer, format="JPEG")  # Use the appropriate format (e.g., JPEG, PNG)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # add string formatting required by the endpoint
        image_string = f"data:image/jpeg;base64,{base64_image}"
        return image_string

    @register_tool
    async def image_qa(self, image_path: str, question: str | None = None) -> str:
        """Generate textual description or answer questions about attached image.

        Args:
            image_path (str): Local path or URL to an image.
            question (str, optional): The question to answer. If not provided, return a description of the image.
        """
        image_str = self._load_image(image_path)
        if not question:
            messages = [
                {"role": "system", "content": TOOL_PROMPTS["image_summary"]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_str}}]},
            ]
            output = await self.llm.query_one(messages=messages, **self.config.config_llm.model_params.model_dump())
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"
        else:
            messages = [
                {"role": "system", "content": TOOL_PROMPTS["image_qa"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_str}},
                    ],
                },
            ]
            output = await self.llm.query_one(messages=messages, **self.config.config_llm.model_params.model_dump())
        return output
