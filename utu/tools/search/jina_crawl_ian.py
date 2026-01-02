import aiohttp
import os
from ...utils import EnvUtils, async_file_cache, get_logger
# from utu.utils import EnvUtils, async_file_cache, get_logger


logger = get_logger(__name__)



class JinaCrawl:
    def __init__(self, config: dict = None) -> None:
        # self.jina_url = "https://r.jina.ai/"
        self.jina_url = "https://ms-knzrtz8l-100034032793-sw.gw.ap-zhongwei.ti.tencentcs.com/ms-knzrtz8l/jina"
        self.jina_header = {}
        # api_key = EnvUtils.get_env("JINA_API_KEY", "")

        USERNAME = os.getenv("HACKER_SEARCH_USERNAME")
        USERID = os.getenv("HACKER_SEARCH_USERID")
        USERTOKEN = os.getenv("HACKER_SEARCH_TOKEN")
        API_KEY = f"{USERNAME}:{USERID}:{USERTOKEN}"

        self.jina_header["Authorization"] = f"Bearer {API_KEY}"
        self.jina_header["Content-Type"] = "application/json"
        self.jina_header["X-With-Generated-Alt"] = "true"
        self.jina_header["X-With-Links-Summary"] = "true"
        self.jina_header["UTU-Use-Cache"] = "true"

        config = config or {}
        # Default timeout: 600 seconds (10 minutes), can be configured via config
        self.timeout = config.get("jina_timeout", 600)

    async def crawl(self, url: str) -> str:
        """standard crawl interface."""
        return await self.crawl_jina(url)

    @async_file_cache(cache_dir="/cfs_turbo/bobshunli/jina_cache", expire_time=None, mode="file")
    async def crawl_jina(self, url: str) -> str:
        # Get the content of the url
        data = {
            "url": url
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(self.jina_url, headers=self.jina_header, json=data) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ValueError(f"Failed to crawl {url} with Jina. Error: {error_text}")

                result = await response.json()
                # Jina 路由现在默认只返回 model_output 内容：
                # - 通常是 str（Markdown/纯文本）
                # - 也可能是 dict/list（当后端返回 JSON 字符串时会被解析）
                if isinstance(result, (dict, list)):
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    print(result)
                # 检查是否有缓存信息
                if 'X-Cache-Hit' in response.headers:
                    print(f"🤖 \n缓存命中: {response.headers['X-Cache-Hit']}")
                return result



if __name__ == "__main__":
    import json
    import asyncio

    jc = JinaCrawl()
    # url = "https://www.baidu.com/"
    url = "https://www.sge.com.cn/sjzx/quotation_daily_new"
    res = asyncio.run(jc.crawl_jina(url))
    print(res)

