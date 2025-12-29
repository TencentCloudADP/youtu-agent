import aiohttp
from bs4 import BeautifulSoup

from ...utils import get_logger
from ..utils import ContentFilter

logger = get_logger(__name__)


class BaiduSearch:
    """Baidu Search."""

    def __init__(self, config: dict = None) -> None:
        self.url = "https://www.baidu.com/s"
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.baidu.com",
        }
        config = config or {}
        search_banned_sites = config.get("search_banned_sites", [])
        self.content_filter = ContentFilter(search_banned_sites) if search_banned_sites else None

    async def search(self, query: str, num_results: int = 5) -> str:
        """standard search interface."""
        res = await self.search_baidu(query)
        # filter
        if self.content_filter:
            results = self.content_filter.filter_results(res["data"], num_results, key="url")
        else:
            results = res["data"][:num_results]
        # format
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append(f"{i}. {r['title']} ({r['url']})")
            if "description" in r:
                formatted_results[-1] += f"\ndescription: {r['description']}"
        msg = "\n".join(formatted_results)
        return msg

    # @async_file_cache(expire_time=None)
    async def search_baidu(self, query: str) -> dict:
        """Search Baidu using web scraping to retrieve relevant search results.

        - WARNING: Uses web scraping which may be subject to rate limiting or anti-bot measures.

        Returns:
            Example result:
            {
                'result_id': 1,
                'title': '百度百科',
                'description': '百度百科是一部内容开放、自由的网络百科全书...',
                'url': 'https://baike.baidu.com/'
            }
        """
        params = {"wd": query, "rn": "20"}
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url, headers=self.headers, params=params) as response:
                response.raise_for_status()  # avoid cache error!
                results = await response.text(encoding="utf-8")

        soup = BeautifulSoup(results, "html.parser")
        results = []
        for idx, item in enumerate(soup.select(".result"), 1):
            title_element = item.select_one("h3 > a")
            title = title_element.get_text(strip=True) if title_element else ""
            link = title_element["href"] if title_element else ""
            desc_element = item.select_one(".c-abstract, .c-span-last")
            desc = desc_element.get_text(strip=True) if desc_element else ""

            results.append(
                {
                    "result_id": idx,
                    "title": title,
                    "description": desc,
                    "url": link,
                }
            )
        if len(results) == 0:
            logger.warning(f"No results found from Baidu search: {query}")
        return {"data": results}
