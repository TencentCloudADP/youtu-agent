import json

import pytest

from utu.config import ConfigLoader
from utu.tools import SearchToolkit
from utu.tools.search.crawl4ai_crawl import Crawl4aiCrawl
from utu.tools.search.google_search import GoogleSearch
from utu.tools.search.jina_crawl import JinaCrawl
from utu.tools.search.jina_search import JinaSearch


# ----------------------------------------------------------------------------
async def test_baidu_search():
    from utu.tools.search.baidu_search import BaiduSearch

    baidu_search = BaiduSearch()
    result = await baidu_search.search_baidu("上海天气")
    print(result)


async def test_google_search():
    google_search = GoogleSearch()
    result = await google_search.search_google("上海天气")
    print(result)


async def test_jina_search():
    jina_search = JinaSearch()
    result = await jina_search.search_jina("明天上海天气")
    print(result)


async def test_duckduckgo_search():
    from utu.tools.search.duckduckgo_search import DuckDuckGoSearch

    duckduckgo_search = DuckDuckGoSearch()
    result = await duckduckgo_search.search_duckduckgo("明天上海天气")
    print(result)


# ----------------------------------------------------------------------------
@pytest.fixture
def search_toolkit() -> SearchToolkit:
    config = ConfigLoader.load_toolkit_config("search")
    return SearchToolkit(config=config)


async def test_tool_schema(search_toolkit: SearchToolkit):
    tools = search_toolkit.get_tools_in_agents()
    for tool in tools:
        print(f"{tool.name}: {tool.description}")
        print(json.dumps(tool.params_json_schema, indent=2, ensure_ascii=False))


# test filter of huggingface.co
# TEST_QUERY = "Illuvium Zero testnet launch date iOS Google Play"
TEST_QUERY = "南京工业大学计算机与信息工程学院 更名 报道"


async def test_search(search_toolkit: SearchToolkit):
    result = await search_toolkit.search(TEST_QUERY, num_results=10)
    print(result)


# ----------------------------------------------------------------------------
TEST_URL = "https://docs.crawl4ai.com/core/simple-crawling/"
TEST_URL = "https://github.com/TencentCloudADP/youtu-agent"
TEST_URL = "https://m.weibo.cn/"


async def test_jina_crawl():
    jina_crawl = JinaCrawl(config={"crawl_params": {}})
    result = await jina_crawl.crawl(TEST_URL)
    print(f"result: {result}")
    jina_crawl = JinaCrawl(config={"crawl_params": {"add_image_desc": True, "add_links": True}})
    result = await jina_crawl.crawl(TEST_URL)
    print(f"result: {result}")


async def test_crawl4ai_crawl():
    crawl4ai_crawl = Crawl4aiCrawl()
    result = await crawl4ai_crawl.crawl(TEST_URL)
    print(result)


# ----------------------------------------------------------------------------
queries = (
    ("https://github.com/TencentCloudADP/Youtu-agent", ""),
    ("https://docs.crawl4ai.com/core/simple-crawling/", "How to log?"),
    ("https://github.com/theskumar/python-dotenv", "Summary this page"),
)


async def test_web_qa(search_toolkit: SearchToolkit):
    for q in queries:
        print(f"query: {q}")
        result = await search_toolkit.web_qa(*q)
        print(f"result: {result}")
