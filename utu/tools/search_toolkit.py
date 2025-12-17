import asyncio

from ..config import ToolkitConfig
from ..utils import SimplifiedAsyncOpenAI, get_logger, oneline_object
from .base import TOOL_PROMPTS, AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class SearchToolkit(AsyncBaseToolkit):
    """Search Toolkit

    NOTE:
        - Please configure the required env variables! See `configs/agents/tools/search.yaml`

    Methods:
        - search(query: str, num_results: int = 5)
        - web_qa(url: str, query: str)
    """

    def __init__(self, config: ToolkitConfig = None):
        super().__init__(config)
        search_engine = self.config.config.get("search_engine", "google")
        match search_engine:
            case "google":
                from .search.google_search import GoogleSearch

                self.search_engine = GoogleSearch(self.config.config)
            case "jina":
                from .search.jina_search import JinaSearch

                self.search_engine = JinaSearch(self.config.config)
            case "baidu":
                from .search.baidu_search import BaiduSearch

                self.search_engine = BaiduSearch(self.config.config)
            case "duckduckgo":
                from .search.duckduckgo_search import DuckDuckGoSearch

                self.search_engine = DuckDuckGoSearch(self.config.config)
            case _:
                raise ValueError(f"Unsupported search engine: {search_engine}")
        crawl_engine = self.config.config.get("crawl_engine", "jina")
        match crawl_engine:
            case "jina":
                from .search.jina_crawl import JinaCrawl

                self.crawl_engine = JinaCrawl(self.config.config)
            case "crawl4ai":
                from .search.crawl4ai_crawl import Crawl4aiCrawl

                self.crawl_engine = Crawl4aiCrawl(self.config.config)
            case _:
                raise ValueError(f"Unsupported crawl engine: {crawl_engine}")
        # llm for web_qa
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )
        self.summary_token_limit = self.config.config.get("summary_token_limit", 1_000)

    @register_tool
    async def search(self, query: str, num_results: int = 5) -> dict:
        """web search to gather information from the web.

        Tips:
        1. search query should be concrete and not vague or super long
        2. try to add Google search operators in query if necessary,
        - " " for exact match;
        - -xxx for exclude;
        - * wildcard matching;
        - filetype:xxx for file types;
        - site:xxx for site search;
        - before:YYYY-MM-DD, after:YYYY-MM-DD for time range.

        Args:
            query (str): The query to search for.
            num_results (int, optional): The number of results to return. Defaults to 5.
        """
        # https://serper.dev/playground
        logger.info(f"[tool] search: {oneline_object(query)}")
        res = await self.search_engine.search(query, num_results)
        logger.info(oneline_object(res))
        return res

    @register_tool
    async def search_web(self, query: str, num_results: int = 5) -> dict:
        """web search to gather information from the web.

        Tips:
        1. search query should be concrete and not vague or super long
        2. try to add Google search operators in query if necessary,
        - " " for exact match;
        - -xxx for exclude;
        - * wildcard matching;
        - filetype:xxx for file types;
        - site:xxx for site search;
        - before:YYYY-MM-DD, after:YYYY-MM-DD for time range.

        Args:
            query (str): The query to search for.
            num_results (int, optional): The number of results to return. Defaults to 5.
        """
        # https://serper.dev/playground
        logger.info(f"[tool] search: {oneline_object(query)}")
        res = await self.search_engine.search(query, num_results)
        logger.info(oneline_object(res))
        return res

    @register_tool
    async def web_qa(self, url: str, query: str) -> str:
        """Ask question to a webpage, you will get the answer and related links from the specified url.

        Tips:
        - Use cases: gather information from a webpage, ask detailed questions.

        Args:
            url (str): The url to ask question to.
            query (str): The question to ask. Should be clear, concise, and specific.
        """
        logger.info(f"[tool] web_qa: {oneline_object({url, query})}")
        content = await self.crawl_engine.crawl(url)
        query = (
            query or "Summarize the content of this webpage, in the same language as the webpage."
        )  # use the same language
        res_summary, res_links = await asyncio.gather(
            self._qa(content, query), self._extract_links(url, content, query)
        )
        result = f"Summary: {res_summary}\n\nRelated Links: {res_links}"
        return result


    async def _qa(self, content: str, query: str) -> str:
        template = TOOL_PROMPTS["search_qa"].format(content=content, query=query)
        return await self.llm.query_one(
            messages=[{"role": "user", "content": template}], **self.config.config_llm.model_params.model_dump()
        )


    async def _extract_links(self, url: str, content: str, query: str) -> str:
        template = TOOL_PROMPTS["search_related"].format(url=url, content=content, query=query)
        return await self.llm.query_one(
            messages=[{"role": "user", "content": template}], **self.config.config_llm.model_params.model_dump()
        )

