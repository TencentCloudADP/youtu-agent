try:
    from ddgs import DDGS
except ImportError as e:
    raise ImportError("Please install ddgs first: `uv pip install ddgs`") from e
from ...utils import get_logger
from ..utils import ContentFilter

logger = get_logger(__name__)


class DuckDuckGoSearch:
    """DuckDuckGo Search.

    - repo: https://github.com/deedy5/ddgs
    """

    def __init__(self, config: dict = None) -> None:
        self.ddgs = DDGS()
        config = config or {}
        search_banned_sites = config.get("search_banned_sites", [])
        self.content_filter = ContentFilter(search_banned_sites) if search_banned_sites else None

    async def search(self, query: str, num_results: int = 5) -> str:
        """standard search interface."""
        res = await self.search_duckduckgo(query)
        # filter
        if self.content_filter:
            results = self.content_filter.filter_results(res, num_results, key="href")
        else:
            results = res[:num_results]
        # format
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append(f"{i}. {r['title']} ({r['href']})")
            if "body" in r:
                formatted_results[-1] += f"\nbody: {r['body']}"
        msg = "\n".join(formatted_results)
        return msg

    async def search_duckduckgo(self, query: str) -> list:
        """Use DuckDuckGo search engine to search for information on the given query.

        Returns:
            [{
                "title": ...
                "href": ...
                "body": ...
            }]
        """
        results = self.ddgs.text(query, max_results=100)
        return results
