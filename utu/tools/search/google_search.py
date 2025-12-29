import aiohttp
import os
import json

from ...utils import EnvUtils, async_file_cache, get_logger
from ..utils import ContentFilter

logger = get_logger(__name__)


class GoogleSearch:
    """Google Search.

    - API key: `SERPER_API_KEY`
    """

    def __init__(self, config: dict = None) -> None:
        use_local_search = os.getenv("USE_LOCAL_SEARCH", "false").lower()
        self.use_local = use_local_search == "true"
        self.use_hacker = use_local_search == "hacker"
        
        if self.use_hacker:
            # Use hacker API endpoint
            self.serper_url = os.getenv(
                "HACKER_SEARCH_URL",
                "http://43.163.181.150:8000/trpc.youtu.llm_interface_service.Greeter/DescribeLlmResult"
            )
            self.serper_header = {
                "Content-Type": "application/json",
                "User-Agent": "ifbook-http-client"
            }
            # Get sec_info from environment variables or use defaults
            self.sec_info = {
                "username": os.getenv("HACKER_SEARCH_USERNAME"),
                "userid": os.getenv("HACKER_SEARCH_USERID"),
                "token": os.getenv("HACKER_SEARCH_TOKEN")
            }
        elif self.use_local:
            # self.serper_url = "https://ms-tpbjbs5p-100034032793-sw.gw.ap-shanghai.ti.tencentcs.com/ms-tpbjbs5p/search"
            # self.serper_header = {"Content-Type": "application/json"}
            self.serper_url = "https://ms-knzrtz8l-100034032793-sw.gw.ap-zhongwei.ti.tencentcs.com/ms-knzrtz8l/serper_search"
            USERNAME = os.getenv("HACKER_SEARCH_USERNAME")
            USERID = os.getenv("HACKER_SEARCH_USERID")
            USERTOKEN = os.getenv("HACKER_SEARCH_TOKEN")
            API_KEY = f"{USERNAME}:{USERID}:{USERTOKEN}"
            print(">>> API_KEY", API_KEY)
            self.serper_header = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "UTU-Use-Cache": "true",
                # "UTU-Cache-Max-Age": "7d"
                }
        else:
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

    @async_file_cache(cache_dir="/cfs_turbo/bobshunli/google_cache", expire_time=None, mode="file")
    async def search_google(self, query: str) -> dict:
        """Call the search API and cache the results."""
        max_num = min(self.search_params.get("num", 10), 20)
        
        if self.use_hacker:
            # Hacker API payload
            params = {
                "sec_info": self.sec_info,
                "model_type": "openai",
                "model_name": "serper",
                "params": json.dumps({"q": query, "num": max_num})
            }
        elif self.use_local:
            # Local API payload
            params = {"q": query, "num": max_num}
        else:
            # Serper API payload
            params = {"q": query, **self.search_params, "num": max_num}

        retries = 1
        for attempt in range(retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.serper_url, headers=self.serper_header, json=params) as response:
                        if not response.ok:
                            error_text = await response.text()
                            logger.warning(f"Google Search API error (attempt {attempt+1}): {response.status}, Body: {error_text}")
                            if attempt < retries:
                                continue
                            return {"error": f"Client error: {response.status}, message='{response.reason}', body='{error_text}'", "organic": []}

                        results = await response.json()
                        # print(">>> results", type(results), results)
                        if self.use_hacker:
                            # Hacker API returns:
                            # { request_id, error_code, message, model_output: "<json string of serper response>" }
                            if isinstance(results, dict):
                                error_code = results.get("error_code")
                                if error_code:
                                    msg = results.get("message") or ""
                                    return {"error": f"Hacker search error_code={error_code}, message={msg}", "organic": []}

                                model_output = results.get("model_output")
                                if isinstance(model_output, str) and model_output:
                                    try:
                                        parsed = json.loads(model_output)
                                    except Exception as e:  # noqa: BLE001
                                        return {"error": f"Failed to parse hacker model_output as JSON: {e}", "organic": []}
                                    if isinstance(parsed, dict):
                                        final_result = parsed
                                    elif isinstance(parsed, list):
                                        final_result = {"organic": parsed}
                                    else:
                                        final_result = {"organic": []}
                                elif isinstance(model_output, dict):
                                    final_result = model_output
                                else:
                                    # Fallback compatibility if upstream changes
                                    if "organic" in results:
                                        final_result = results
                                    else:
                                        final_result = {"organic": []}
                            elif isinstance(results, list):
                                final_result = {"organic": results}
                            else:
                                final_result = {"organic": []}
                        
                        elif self.use_local:
                            # Wrap list in expected structure for local service
                            # final_result = {"organic": results['organic']}
                            final_result = results
                        else:
                            final_result = results
                        
                        if final_result.get("organic"):
                            return final_result
                        
                        logger.warning(f"Google Search returned empty results (attempt {attempt+1})")
                        if attempt == retries:
                            return final_result

            except Exception as e:
                logger.warning(f"Exception during Google search (attempt {attempt+1}): {e}")
                if attempt == retries:
                    return {"error": f"Unexpected error: {e}", "organic": []}
        
        return {"organic": []}
