import os
from collections.abc import Callable
from typing import Any

import httpx

from ..config import ToolkitConfig
from .base import AsyncBaseToolkit


class SpecStreamToolkit(AsyncBaseToolkit):
    """
    A toolkit for interacting with the SpecStream API (api.specstream.ai).
    """

    def __init__(self, config: ToolkitConfig | None = None) -> None:
        super().__init__(config)
        self.api_key = os.getenv("SPECSTREAM_API_KEY")
        if not self.api_key:
            raise ValueError("SPECSTREAM_API_KEY environment variable is required")
        self.async_client = httpx.AsyncClient()

    async def _search(self, query: str, num: int = 10) -> dict[str, Any]:
        """
        Helper function to make requests to the SpecStream API.
        """
        url = "https://api.specstream.ai/v1/web-search"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        params = {"q": query, "num": num}
        try:
            response = await self.async_client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error occurred: {e.response.status_code} {e.response.text}"}
        except httpx.RequestError as e:
            return {"error": f"An error occurred: {e}"}

    async def web_search(self, query: str, num: int = 10) -> dict[str, Any]:
        """
        Search the web using SpecStream.

        Args:
            query (str): Search query string.
            num (int): Number of results to return.

        Returns:
            Dict[str, Any]: A dictionary containing the search results.
        """
        result = await self._search(query, num)
        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "results": result.get("results", []),
            "status": "success",
        }

    async def get_tools_map(self) -> dict[str, Callable]:
        return {
            "web_search": self.web_search,
        }

    async def cleanup(self):
        await self.async_client.aclose()
