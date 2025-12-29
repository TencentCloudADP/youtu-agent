import aiohttp

from ...utils import EnvUtils, async_file_cache, get_logger
from ..utils import ContentFilter

logger = get_logger(__name__)


class JinaSearch:
    """Search service provided by Jina

    - API key: `JINA_API_KEY`
    """

    def __init__(self, config: dict = None) -> None:
        self.jina_url = "https://s.jina.ai/"
        self.jina_header = {
            "Accept": "application/json",
            "Authorization": f"Bearer {EnvUtils.get_env('JINA_API_KEY')}",
            "X-Respond-With": "no-content",  # do not return content
        }
        config = config or {}
        self.search_params = config.get("search_params", {})
        search_banned_sites = config.get("search_banned_sites", [])
        self.content_filter = ContentFilter(search_banned_sites) if search_banned_sites else None

    async def search(self, query: str, num_results: int = 5) -> str:
        """standard search interface."""
        res = await self.search_jina(query)
        # filter
        if self.content_filter:
            results = self.content_filter.filter_results(res["data"], num_results, "url")
        else:
            results = res["data"][:num_results]
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append(f"{i}. {r['title']} ({r['url']})")
            if "description" in r:
                formatted_results[-1] += f"\ndescription: {r['description']}"
        msg = "\n".join(formatted_results)
        return msg

    @async_file_cache(expire_time=None)
    async def search_jina(self, query: str) -> dict:
        """Call the Jina API and cache the results.

        Ref: https://jina.ai/api-dashboard

        Returns:
            {
                "data": [{
                    "title": ...
                    "url": ...
                    "description": ...
                    "content": ...
                    "metadata", "external", "usage": ...
                }]
            }
        """
        params = {"q": query, **self.search_params}
        async with aiohttp.ClientSession() as session:
            async with session.get(self.jina_url, headers=self.jina_header, params=params) as response:
                response.raise_for_status()
                results = await response.json()
                return results
