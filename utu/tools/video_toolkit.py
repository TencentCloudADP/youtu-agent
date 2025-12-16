"""
https://github.com/googleapis/python-genai
https://ai.google.dev/gemini-api/docs/api-key
"""

try:
    from google import genai
    from google.genai.types import HttpOptions, Part
except ImportError:
    pass

from ..config import ToolkitConfig
from ..utils import get_logger
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class VideoToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        self.client = genai.Client(
            api_key=self.config.config.get("google_api_key"), http_options=HttpOptions(api_version="v1alpha")
        )
        self.model = self.config.config.get("google_model")

    @register_tool
    async def video_qa(self, video_url: str, question: str) -> str:
        r"""Asks a question about the video.

        Args:
            video_url (str): The path or URL to the video file.
            question (str): The question to ask about the video.
        """
        if not video_url.startswith("http"):
            video_part = Part.from_uri(file_uri=video_url)
        else:
            # e.g. Youtube URL
            video_part = Part.from_uri(
                file_uri=video_url,
                mime_type="video/mp4",
            )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                question,
                video_part,
            ],
        )

        logger.debug(f"Video analysis response from gemini: {response.text}")
        return response.text
