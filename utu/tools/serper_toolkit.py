import os
from collections.abc import Callable
from typing import Any

import httpx

from ..config import ToolkitConfig
from .base import AsyncBaseToolkit, register_tool


class SerperToolkit(AsyncBaseToolkit):
    """
    A toolkit for interacting with the Serper API (google.serper.dev).
    """

    def __init__(self, config: ToolkitConfig | None = None) -> None:
        super().__init__(config)
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY environment variable is required")
        self.async_client = httpx.AsyncClient()

    async def _search(self, endpoint: str, payload: dict) -> dict[str, Any]:
        """
        Helper function to make requests to the Serper API.
        """
        url = f"https://google.serper.dev/{endpoint}"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        try:
            response = await self.async_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error occurred: {e.response.status_code} {e.response.text}"}
        except httpx.RequestError as e:
            return {"error": f"An error occurred: {e}"}

    async def google_search(
        self,
        query: str,
        location: str = "China",
        gl: str = "cn",
        hl: str = "zh-cn",
        num: int = 10,
        date_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using Google Search.

        Args:
            query (str): Search query string.
            location (str): Geographic location for search.
            gl (str): Country code for search results.
            hl (str): Language code for search interface.
            num (int): Number of results to return.
            date_range (str, optional): Time filter for search results.
                Options: "h" (past hour), "d" (past day), "w" (past week),
                "m" (past month), "y" (past year). Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the search results.
        """
        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}
        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._search("search", payload)
        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "date_range": date_range,
            "results": result.get("organic", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("organic", [])),
            "status": "success",
        }

    async def autocomplete(
        self, query: str, location: str = "China", gl: str = "cn", hl: str = "zh-cn"
    ) -> dict[str, Any]:
        """
        Get autocomplete suggestions for a given query.

        Args:
            query (str): The partial query string.
            location (str): Geographic location for search.
            gl (str): Country code for search results.
            hl (str): Language code for search interface.

        Returns:
            Dict[str, Any]: A dictionary containing autocomplete suggestions.
        """
        payload = {"q": query, "location": location, "gl": gl, "hl": hl}
        result = await self._search("autocomplete", payload)
        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "suggestions": result.get("suggestions", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_suggestions": len(result.get("suggestions", [])),
            "status": "success",
        }

    async def google_lens(self, url: str, gl: str = "cn", hl: str = "zh-cn", num: int = 10) -> dict[str, Any]:
        """
        Analyze an image using Google Lens.

        Args:
            url (str): The URL of the image to analyze.
            gl (str): Country code for search results.
            hl (str): Language code for search interface.
            num (int): Number of results to return.

        Returns:
            Dict[str, Any]: A dictionary containing the visual search results.
        """
        payload = {"url": url, "gl": gl, "hl": hl}
        result = await self._search("lens", payload)
        if "error" in result:
            return {"url": url, "error": result["error"], "status": "error"}

        return {
            "url": url,
            "gl": gl,
            "hl": hl,
            "results": result.get("organic", [])[:num],
            "searchParameters": result.get("searchParameters", {}),
            "total_results": min(len(result.get("organic", [])), num),
            "status": "success",
        }

    @register_tool
    async def image_search(
        self,
        query: str,
        location: str = "China",
        gl: str = "cn",
        hl: str = "zh-cn",
        num: int = 10,
        date_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for images using Google Images.

        Args:
            query (str): The search query.
            location (str): Geographic location for search.
            gl (str): Country code for search results.
            hl (str): Language code for search interface.
            num (int): Number of results to return.
            date_range (str, optional): Time filter. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary with image search results.
        """
        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}
        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._search("images", payload)
        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "results": result.get("images", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("images", [])),
            "status": "success",
        }

    async def map_search(
        self,
        query: str,
        hl: str = "zh-cn",
        latitude: float | None = None,
        longitude: float | None = None,
        zoom: int | None = 18,
        place_id: str | None = None,
        cid: str | None = None,
        num: int = 10,
    ) -> dict[str, Any]:
        """
        Search for locations using Google Maps.

        Args:
            query (str): Search query.
            hl (str): Language code.
            latitude (Optional[float]): GPS latitude.
            longitude (Optional[float]): GPS longitude.
            zoom (Optional[int]): Map zoom level.
            place_id (Optional[str]): Google Place ID.
            cid (Optional[str]): Google CID.
            num (int): Number of results.

        Returns:
            Dict[str, Any]: Dictionary with map search results.
        """
        payload = {"q": query, "hl": hl, "num": num}
        if latitude is not None and longitude is not None:
            if zoom is not None:
                payload["ll"] = f"@{latitude},{longitude},{zoom}z"
            else:
                payload["ll"] = f"@{latitude},{longitude}"
        if place_id:
            payload["placeId"] = place_id
        if cid:
            payload["cid"] = cid

        result = await self._search("maps", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "hl": hl,
            "latitude": latitude,
            "longitude": longitude,
            "zoom": zoom,
            "place_id": place_id,
            "cid": cid,
            "results": result.get("places", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("places", [])),
            "status": "success",
        }

    async def news_search(
        self,
        query: str,
        location: str = "China",
        gl: str = "cn",
        hl: str = "zh-cn",
        num: int = 10,
        date_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for news articles using Google News.

        Args:
            query (str): Search query.
            location (str): Geographic location.
            gl (str): Country code.
            hl (str): Language code.
            num (int): Number of results.
            date_range (Optional[str]): Time filter.

        Returns:
            Dict[str, Any]: Dictionary with news search results.
        """
        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}
        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._search("news", payload)
        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "date_range": date_range,
            "results": result.get("news", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("news", [])),
            "status": "success",
        }

    async def place_search(
        self,
        query: str,
        location: str = "China",
        gl: str = "cn",
        hl: str = "zh-cn",
        num: int = 10,
        date_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for places using Google Places.

        Args:
            query (str): Search query.
            location (str): Geographic location.
            gl (str): Country code.
            hl (str): Language code.
            num (int): Number of results.
            date_range (Optional[str]): Time filter.

        Returns:
            Dict[str, Any]: Dictionary with place search results.
        """
        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}
        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._search("places", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "results": result.get("places", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("places", [])),
            "status": "success",
        }

    async def scholar_search(
        self, query: str, location: str = "China", gl: str = "cn", hl: str = "zh-cn", num: int = 10
    ) -> dict[str, Any]:
        """
        Search for academic papers using Google Scholar.

        Args:
            query (str): Search query.
            location (str): Geographic location.
            gl (str): Country code.
            hl (str): Language code.
            num (int): Number of results.

        Returns:
            Dict[str, Any]: Dictionary with scholar search results.
        """
        payload = {
            "q": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "num": num,
        }

        result = await self._search("scholar", payload)
        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "results": result.get("organic", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("organic", [])),
            "status": "success",
        }

    async def video_search(
        self,
        query: str,
        location: str = "China",
        gl: str = "cn",
        hl: str = "zh-cn",
        num: int = 10,
        date_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for videos using Google Videos.

        Args:
            query (str): Search query.
            location (str): Geographic location.
            gl (str): Country code.
            hl (str): Language code.
            num (int): Number of results.
            date_range (Optional[str]): Time filter.

        Returns:
            Dict[str, Any]: Dictionary with video search results.
        """
        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}
        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._search("videos", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "results": result.get("videos", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("videos", [])),
            "status": "success",
        }

    async def get_tools_map(self) -> dict[str, Callable]:
        return {
            "google_search": self.google_search,
            "autocomplete": self.autocomplete,
            "google_lens": self.google_lens,
            "image_search": self.image_search,
            "map_search": self.map_search,
            "news_search": self.news_search,
            "place_search": self.place_search,
            "scholar_search": self.scholar_search,
            "video_search": self.video_search,
        }

    async def cleanup(self):
        await self.async_client.aclose()
