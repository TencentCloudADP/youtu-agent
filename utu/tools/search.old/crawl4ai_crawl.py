try:
    from crawl4ai import AsyncWebCrawler
except ImportError as e:
    raise ImportError(
        "Please install crawl4ai: `uv pip install crawl4ai && python -m playwright install --with-deps chromium`"
    ) from e  # noqa: E501
from ...utils import async_file_cache, get_logger

logger = get_logger(__name__)


class Crawl4aiCrawl:
    """Crawl4ai Crawl.

    - repo: https://github.com/unclecode/crawl4ai
    """

    def __init__(self, config: dict = None) -> None:
        config = config or {}

    async def crawl(self, url: str) -> str:
        """standard crawl interface."""
        return await self.crawl_crawl4ai(url)

    @async_file_cache(expire_time=None)
    async def crawl_crawl4ai(self, url: str) -> str:
        # Get the content of the url
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
            )
            return result.markdown
