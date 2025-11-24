# ruff: noqa: E501
import requests
import json
from utu.tools import AsyncBaseToolkit, register_tool
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch


class WikiToolkit(AsyncBaseToolkit):

    @register_tool
    async def search_wiki(self, query: list[str]) -> str:
        """Performs batched searches on wikipedia: supply an array 'query'; the tool retrieves the top 3 results for each query in one call.
        
        Args:
            query: Array of query strings. Include multiple complementary search queries in a single call.
        """
        # print("query", query)
        retrieval_service_url = "http://10.16.20.181:80/retrieve"  # NOTE: you should change this to your own deployment
        result_text, metadata = perform_single_search_batch(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query,
                    topk=3,
                    concurrent_semaphore=None,  # Ray handles concurrency control
                    timeout=30,
                )
        result_text_str = json.loads(result_text)["result"]
        print(f"💻[wikipedia] >>> {query} searching total returned response: \n\n{result_text_str}")
        return result_text_str

