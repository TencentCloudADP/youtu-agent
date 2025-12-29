import aiohttp

from ...utils import EnvUtils, async_file_cache, get_logger
from ..utils import ContentFilter

logger = get_logger(__name__)


class GoogleSearch:
    """Google Search.

    - API key: `SERPER_API_KEY`
    """

    def __init__(self, config: dict = None) -> None:
        self.serper_url = r"https://google.serper.dev/search"
        self.serper_header = {"X-API-KEY": EnvUtils.get_env("SERPER_API_KEY"), "Content-Type": "application/json"}
        config = config or {}
        self.search_params = config.get("search_params", {})
        search_banned_sites = config.get("search_banned_sites", [])
        self.content_filter = ContentFilter(search_banned_sites) if search_banned_sites else None

    async def search(self, query: str, num_results: int = 5) -> str:
        """standard search interface."""
        res = await self.search_google(query)
        # filter
        if self.content_filter:
            results = self.content_filter.filter_results(res["organic"], num_results)
        else:
            results = res["organic"][:num_results]
        # format
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append(f"{i}. {r['title']} ({r['link']})")
            if "snippet" in r:
                formatted_results[-1] += f"\nsnippet: {r['snippet']}"
            if "sitelinks" in r:
                formatted_results[-1] += f"\nsitelinks: {r['sitelinks']}"
        msg = "\n".join(formatted_results)
        return msg

    @async_file_cache(expire_time=None)
    async def search_google(self, query: str) -> dict:
        """Call the serper.dev API and cache the results."""
        params = {"q": query, **self.search_params, "num": 10}  # fetch and cache the results
        async with aiohttp.ClientSession() as session:
            async with session.post(self.serper_url, headers=self.serper_header, json=params) as response:
                response.raise_for_status()  # avoid cache error!
                results = await response.json()
                return results
