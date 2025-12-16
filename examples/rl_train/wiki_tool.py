# ruff: noqa: E501
import requests

from utu.tools import AsyncBaseToolkit, register_tool


class WikiToolkit(AsyncBaseToolkit):
    @register_tool
    async def perform_single_search_batch(self, query: list[str]) -> str:
        """Performs batched searches on wikipedia: supply an array 'query'; the tool retrieves the top 5 results for each query in one call.

        Args:
            query: Array of query strings. Include multiple complementary search queries in a single call.
        """
        payload = {"queries": query, "topk": 5, "return_scores": True}

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(
            "http://10.0.20.2:8501/retrieve",  # NOTE: you should change this to your own deployment
            headers=headers,
            json=payload,
            timeout=30,
        )
        api_response = response.json()
        raw_results = api_response.get("result", [])
        if raw_results:
            assert len(query) == len(raw_results)
            pretty_results = []
            total_results = 0
            for query_item, retrieval in zip(query, raw_results, strict=False):
                formatted = self._passages2string(retrieval)
                formatted = f'🔍  Results for query "{query_item}"\n{formatted}'
                pretty_results.append(formatted)
                total_results += len(retrieval) if isinstance(retrieval, list) else 1
            final_result = "\n================\n".join(pretty_results)
        return final_result

    def _passages2string(self, retrieval_result):
        """Convert retrieval results to formatted string."""
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
        return format_reference.strip()
