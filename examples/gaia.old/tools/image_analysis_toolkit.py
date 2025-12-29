# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

# ruff: noqa

import base64
from collections.abc import Callable
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image

from utu.config import ToolkitConfig
from utu.tools import AsyncBaseToolkit
from utu.utils import EnvUtils, SimplifiedAsyncOpenAI, get_logger

logger = get_logger(__name__)


class ImageAnalysisToolkit(AsyncBaseToolkit):
    r"""A toolkit for comprehensive image analysis and understanding.
    The toolkit uses vision-capable language models to perform these tasks.
    """

    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        image_llm_config = {
            "type": EnvUtils.get_env("UTU_IMAGE_LLM_TYPE", "chat.completions"),
            "model": EnvUtils.get_env("UTU_IMAGE_LLM_MODEL"),
            "api_key": EnvUtils.get_env("UTU_IMAGE_LLM_API_KEY"),
            "base_url": EnvUtils.get_env("UTU_IMAGE_LLM_BASE_URL"),
        }
        self.llm = SimplifiedAsyncOpenAI(**image_llm_config)

    async def image_to_text(self, image_path: str) -> str:
        r"""Generates textual description of an image with optional custom
        prompt.

        Args:
            image_path (str): Local path or URL to an image file.

        Returns:
            str: Natural language description of the image.
        """
        system_msg = """You are an image analysis expert. Provide a detailed description including text if present."""

        return await self._analyze_image(
            image_path=image_path,
            prompt="Please describe the contents of this image.",
            system_message=system_msg,
        )

    async def ask_question_about_image(self, image_path: str, question: str) -> str:
        r"""Answers image questions with optional custom instructions.

        Args:
            image_path (str): Local path or URL to an image file.
            question (str): Query about the image content.

        Returns:
            str: Detailed answer based on visual understanding
        """
        logger.info(f"Calling image analysis toolkit with question: {question} and image path: {image_path}")
        system_msg = """Answer questions about images by:
            1. Careful visual inspection
            2. Contextual reasoning
            3. Text transcription where relevant
            4. Logical deduction from visual evidence"""

        return await self._analyze_image(
            image_path=image_path,
            prompt=question,
            system_message=system_msg,
        )

    def _load_image(self, image_path: str) -> str:
        parsed = urlparse(image_path)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        }

        if parsed.scheme in ("http", "https"):
            logger.debug(f"Fetching image from URL: {image_path}")
            response = requests.get(image_path, timeout=15, headers=headers)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            logger.debug(f"Loading local image: {image_path}")
            img = Image.open(image_path).convert("RGB")

        buffer = BytesIO()
        img.save(buffer, format="JPEG")  # Use the appropriate format (e.g., JPEG, PNG)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_string = f"data:image/jpeg;base64,{base64_image}"
        return image_string

    async def _analyze_image(self, image_path: str, prompt: str, system_message: str) -> str:
        try:
            image = self._load_image(image_path)
            logger.info(f"Analyzing image: {image_path}")

            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image}}],
                },
            ]

            response = await self.llm.query_one(messages=messages, **self.config.config_llm.model_params.model_dump())
            return response

        except (ValueError, requests.exceptions.RequestException) as e:
            logger.error(f"Image handling error: {e}")
            return f"Image error: {e!s}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Analysis failed: {e!s}"

    async def get_tools_map(self) -> dict[str, Callable]:
        return {
            "image_to_text": self.image_to_text,
            "ask_question_about_image": self.ask_question_about_image,
        }
