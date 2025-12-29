# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

# ruff: noqa: E501

import asyncio
import os
import pathlib
import traceback
from typing import Any

from utu.config import ToolkitConfig
from utu.tools import AsyncBaseToolkit, register_tool
from utu.tools.search.google_search import GoogleSearch
from utu.tools.search.duckduckgo_search import DuckDuckGoSearch
from utu.tools.search.jina_crawl import JinaCrawl
from utu.utils import FileUtils, SimplifiedAsyncOpenAI, get_logger, TokenUtils



logger = get_logger(__name__)
PROMPTS = FileUtils.load_prompts(pathlib.Path(__file__).parent / "search_prompts.yaml")


class SearchToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)

        self.crawler = JinaCrawl()
        if self.config.config_llm:
            model_config = self.config.config_llm.model_provider.model_dump()
        else:
            model_config = {}
        # ---------------------------------------------------- 定制化Bob工具使用默认的qwen3-4b ----------------------------------------------------- #
        model_config["base_url"] = "https://ms-d2chnwst-100034032793-sw.gw.ap-zhongwei.ti.tencentcs.com/ms-d2chnwst/v1"
        self.llm = SimplifiedAsyncOpenAI(
            **model_config if self.config.config_llm else {}
        )

        config = {
            # "search_params": {
            #     "hl": "en",  # language code
            #     "page": 1,  # page number
            # }
            "search_banned_sites": [
                "https://huggingface.co/",
                "https://grok.com/share/",
                "https://modelscope.cn/datasets/",
            ]
        }
        self.google_searcher = GoogleSearch(config)
        self.duckduckgo_searcher = DuckDuckGoSearch(config)
        self.max_concurrent_jina_calls = 3
        self.cache_dir = "workspace/"
        os.makedirs(self.cache_dir, exist_ok=True)
        # Token limit for content to avoid exceeding model's max context length
        # Default: 25000 tokens (leaving ~7000 tokens for prompt and output, assuming 32k context)
        self.max_content_tokens = self.config.config.get("max_content_tokens", 28000)

    @register_tool
    async def search_google(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        r"""Use Google search engine to search information for the given query.

        Args:
            query (str): The query to be searched.
            num_results (int): The number of search results to retrieve, should better be larger than 5.

        Returns:
            List[Dict[str, Any]): A list of dictionaries where each dictionary
            represents a website.
                Each dictionary contains the following keys:
                - 'result_id': A number in order.
                - 'title': The title of the website.
                - 'description': A brief description of the website.
                - 'long_description': More detail of the website.
                - 'url': The URL of the website.

                Example:
                {
                    'result_id': 1,
                    'title': 'OpenAI',
                    'description': 'An organization focused on ensuring that
                    artificial general intelligence benefits all of humanity.',
                    'long_description': 'OpenAI is a non-profit artificial
                    intelligence research company. Our goal is to advance
                    digital intelligence in the way that is most likely to
                    benefit humanity as a whole',
                    'url': 'https://www.openai.com'
                }
            title, description, url of a website.
        """
        try:
            response = await self.google_searcher.search_google(query)
            # response = await self.duckduckgo_searcher.search_duckduckgo_googlestyle(query)
            
            # Check if response is valid
            if not isinstance(response, dict):
                logger.warning(f"Unexpected response type from Google Search: {type(response)}. Query: {query[:100]}")
                return []
            
            # Check for API errors
            if "error" in response:
                error_msg = response.get("error", "Unknown error")
                logger.warning(f"Google Search API returned error: {error_msg}. Query: {query[:100]}")
                return []
            
            # Check if organic results exist
            if "organic" not in response:
                logger.warning(f"No 'organic' field in Google Search response. Keys: {list(response.keys())}. Query: {query[:100]}")
                # Log full response for debugging (truncated)
                logger.debug(f"Full response (truncated): {str(response)[:500]}")
                return []
            
            organic_results = response.get("organic", [])
            if not organic_results:
                logger.debug(f"Google Search returned empty organic results. Query: {query[:100]}")
                return []
            
            # Filter out huggingface.co and gaia URLs
            filtered_results = []
            for result in organic_results:
                if "error" in result:
                    continue
                url = result.get("link", result.get("url", result.get("href", "")))
                # Skip huggingface.co and gaia URLs
                if url and ("huggingface" in url.lower() or "gaia" in url.lower()):
                    logger.debug(f"Filtering out huggingface or gaia URL: {url}")
                    continue
                filtered_results.append(result)
                # Stop when we have enough results
                if len(filtered_results) >= num_results:
                    break
            
            logger.debug(f"Google Search returned {len(filtered_results)} results (after filtering) for query: {query[:100]}")
            return filtered_results
            
        except KeyError as e:
            logger.error(f"Missing key in Google Search response: {e}. Query: {query[:100]}")
            return []
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            # Provide more specific error messages for connection issues
            if "ClientConnectorError" in error_type or "Cannot connect" in error_msg:
                logger.error(
                    f"Connection error to google.serper.dev - cannot reach the search service. "
                    f"This may be due to network issues, firewall restrictions, or service downtime. "
                    f"Query: {query[:100]}"
                )
            else:
                logger.info(f"Traceback: {traceback.format_exc()}")
                logger.error(f"Error during Google search: {error_type}: {error_msg}. Query: {query[:100]}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []

    def _format_web_content(self, query: str, extracted_contents: list[dict[str, str]]) -> str:
        """格式化网页内容搜索结果"""
        if not extracted_contents:
            return f"<search results for '{query}'>\nNo valid content extracted.\n</search results end>"

        formatted_content = f"<search results for '{query}'>\n"

        for i, content_item in enumerate(extracted_contents):
            formatted_content += f"[web_{i + 1}]\n"
            formatted_content += f"web title: {content_item['title']}\n"
            formatted_content += f"web url: {content_item['url']}\n"
            formatted_content += (
                f"web content: {content_item['content'][:5000]}...\n"  # Limit to 5000 chars for better readability
            )
            if i < len(extracted_contents) - 1:
                formatted_content += "\n"

        formatted_content += "\n</search results end>"
        return formatted_content

    @register_tool
    async def multi_query_deep_search(self, search_queries: list[str], question: str) -> str:
        r"""Perform multi-query deep search by using multiple search queries to find comprehensive information about the same topic, then extract webpage content and use LLM analysis.

        Args:
            search_queries (List[str]): A list of different search queries to use for finding information about the same topic.
            question (str): The specific question to ask about all the search results.

        Returns:
            str: The LLM-generated answer to the question based on the extracted webpage content from all queries.
        """
        deep_info_page = True
        add_citation = True

        logger.debug(f"Performing multi-query deep search with {len(search_queries)} queries for question: {question}")

        # Step 1: Perform Google search for each query and extract webpage content
        async def process_single_query(search_query):
            logger.debug(f"Processing search query: {search_query}")

            # Perform Google search
            search_results = await self.search_google(search_query, 10)
            if not search_results or (len(search_results) == 1 and "error" in search_results[0]):
                logger.warning(f"Failed to get search results for query: {search_query}. Search results: {search_results}")
                return []

            # Filter out huggingface.co results and keep max 5
            filtered_results = []
            for result in search_results:
                if "error" in result:
                    continue
                url = result.get("link", result.get("url", ""))
                # Skip huggingface.co URLs
                if url and ("huggingface" in url.lower() or "gaia" in url.lower()):
                    logger.debug(f"Filtering out huggingface or gaia URL: {url}")
                    continue
                filtered_results.append(result)
                # Stop when we have 5 valid results
                if len(filtered_results) >= 3:
                    break

            if not filtered_results:
                logger.warning(f"No valid search results found after filtering for query: {search_query}")
                return []

            logger.debug(f"Using {len(filtered_results)} search results for query: {search_query}")

            # Extract content from each webpage using JINA API
            async def extract_content_from_result(result):
                """Helper function to extract content from a single search result"""
                url = result.get("link", result.get("url", ""))
                if not url:
                    return None

                logger.debug(f"Extracting content from URL: {url}")
                content = await self.crawler.crawl_jina(url)
                if content and not content.startswith("Error while extracting"):
                    return {
                        "url": url,
                        "title": result.get("title", ""),
                        "content": content
                        if len(content) < 100000
                        else content[:100000] + f"...({len(content) - 100000} characters truncated)",
                        "search_query": search_query,
                    }
                return None

            # Use ThreadPoolExecutor for parallel content extraction within this query
            query_extracted_contents = []  # 提取的内容

            async def task(result):
                async with asyncio.Semaphore(5):
                    res = await extract_content_from_result(result)
                    query_extracted_contents.append(res)

            await asyncio.gather(*[task(result) for result in filtered_results])
            logger.debug(
                f"Successfully extracted content from {len(query_extracted_contents)} URLs for query: {search_query}"
            )
            return query_extracted_contents

        all_extracted_contents = []

        async def task(query) -> None:
            async with asyncio.Semaphore(3):
                res = await process_single_query(query)
                all_extracted_contents.extend(res)

        await asyncio.gather(*[task(query) for query in search_queries])

        # URL去重 - 移除重复的URL内容（所有查询的结果去重）
        seen_urls = set()
        unique_all_extracted_contents = []
        for content_item in all_extracted_contents:
            url = content_item["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                unique_all_extracted_contents.append(content_item)
            else:
                logger.debug(f"Skipping duplicate URL across all queries: {url}")

        all_extracted_contents = unique_all_extracted_contents
        if not all_extracted_contents:
            return "Failed to extract content from any of the search results across all queries."
        logger.debug(f"Total extracted content from {len(all_extracted_contents)} unique URLs across all queries")

        # Step 2: Use LLM to analyze the content and answer the question

        # If deep_info_page is enabled, summarize each page individually first
        if deep_info_page:
            logger.debug("deep_info_page enabled, summarizing each page individually")
            page_summaries = []

            async def summarize_single_page(content_item, index):
                """Helper function to summarize a single webpage"""
                # Truncate content to avoid exceeding model's max context length
                content = TokenUtils.truncate_text_by_token(
                    content_item["content"], limit=self.max_content_tokens
                )
                page_prompt = PROMPTS["DEEP_EXTRACTION"].format(
                    query=content_item["search_query"],
                    title=content_item["title"],
                    url=content_item["url"],
                    content=content,
                    question=question,
                )
                messages = [
                    {"role": "system", "content": PROMPTS["SP_SUMMARY"]},
                    {"role": "user", "content": page_prompt},
                ]
                logger.debug(f"Summarizing page {index + 1}: {content_item['url']}")
                try:
                    # page_summary = await self.llm.query_one(messages=messages)
                    # ---------------------------------------------------- 定制化Bob工具使用默认的qwen3-4b ----------------------------------------------------- #
                    page_summary = await self.llm.query_one(messages=messages, model="Qwen3-4B")
                    return {
                        "url": content_item["url"],
                        "title": content_item["title"],
                        "search_query": content_item["search_query"],
                        "summary": page_summary,
                        "index": index,  # Keep track of original order
                    }
                except Exception as e:
                    logger.error(f"Error summarizing page {index + 1} (URL: {content_item['url']}): {type(e).__name__}: {e}")
                    raise

            # Use asyncio.gather for parallel page summarization with concurrency control
            # Limit concurrent LLM requests to avoid connection errors
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent LLM requests
            
            async def task_with_semaphore(content_item, index):
                async with semaphore:
                    return await summarize_single_page(content_item, index)
            
            tasks = [task_with_semaphore(content_item, i) for i, content_item in enumerate(all_extracted_contents)]

            # Execute all summarization tasks concurrently
            temp_summaries = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and sort by original index
            page_summaries = []
            for summary_result in temp_summaries:
                if isinstance(summary_result, Exception):
                    # Error already logged in summarize_single_page, just skip
                    pass
                elif summary_result is not None:
                    page_summaries.append(summary_result)

            # Sort by original index to maintain order
            page_summaries.sort(key=lambda x: x["index"])
            page_summaries = [{k: v for k, v in summary.items() if k != "index"} for summary in page_summaries]

            # Prepare content from summaries for final analysis
            formatted_content = f"<search results for '{question}'>\n"

            # Group summaries by search query
            query_groups = {}
            for summary_item in page_summaries:
                query = summary_item["search_query"]
                if query not in query_groups:
                    query_groups[query] = []
                query_groups[query].append(summary_item)

            # Format grouped results
            for query, summaries in query_groups.items():
                formatted_content += f"<search_query: '{query}'>\n"
                for i, summary_item in enumerate(summaries):
                    formatted_content += f"[web_{i + 1}]\n"
                    formatted_content += f"web title: {summary_item['title']}\n"
                    formatted_content += f"web url: {summary_item['url']}\n"
                    formatted_content += f"web summary: {summary_item['summary']}\n"
                    if i < len(summaries) - 1:
                        formatted_content += "\n"
                formatted_content += "</search_query>\n\n"

            formatted_content += "</search results end>"

        else:
            # Use the original formatting function to prepare content
            formatted_content = self._format_web_content(", ".join(search_queries), all_extracted_contents)

        prompt = PROMPTS["DEEP_FINAL"].format(
            queries=", ".join(search_queries), formatted_content=formatted_content, question=question
        )
        messages = [{"role": "system", "content": PROMPTS["SP_ANALYSIS"]}, {"role": "user", "content": prompt}]
        logger.debug("Sending prompt to LLM for final analysis")
        # final_answer = await self.llm.query_one(messages=messages)
        # ---------------------------------------------------- 定制化Bob工具使用默认的qwen3-4b ----------------------------------------------------- #
        final_answer = await self.llm.query_one(messages=messages, model="Qwen3-4B")

        # Add citations if add_citation is enabled
        if add_citation:
            logger.debug("add_citation enabled, appending citations")
            citations = "\n\n## References:\n"
            for i, content_item in enumerate(all_extracted_contents):
                citations += f"[{i + 1}] {content_item['title']} - {content_item['url']} (Query: {content_item['search_query']})\n"
            final_answer += citations

        return final_answer

    @register_tool
    async def single_query_deep_search(self, search_queries: list[str], question: str) -> str:
        r"""Perform single-query deep search by combining multiple search queries into one comprehensive query to find information about a topic, then extract webpage content and use LLM analysis.

        Args:
            search_queries (List[str]): One or several search queries, depending on the intended breadth of the search, to be combined into a single comprehensive search.
            question (str): The specific question to ask about the search results.

        Returns:
            str: The LLM-generated answer to the question based on the extracted webpage content.
        """
        deep_info_page = True
        add_citation = True

        # Step 1: Combine multiple queries into a single comprehensive query
        if not search_queries:
            return "No search queries provided."

        if len(search_queries) == 1:
            combined_query = search_queries[0]
        else:
            # Combine queries with OR logic for broader search
            # Use OR without quotes for better results (tested: quotes cause 0 results for long queries)
            # Limit query length to avoid API issues (max ~200 chars works better)
            max_query_length = 200
            combined_query = " OR ".join(search_queries)
            
            # If combined query is too long, try without quotes first, then fallback to individual queries
            if len(combined_query) > max_query_length:
                logger.debug(f"Combined query too long ({len(combined_query)} chars), using first query as fallback")
                combined_query = search_queries[0]  # Use first query if too long

        logger.debug(f"Performing single comprehensive search with combined query: {combined_query}")

        # Step 2: Perform Google search with the combined query
        search_results = await self.search_google(combined_query, 15)  # Use more results since we're doing one search

        # Fallback mechanism: if combined query returns 0 results, try individual queries
        if not search_results or (len(search_results) == 1 and "error" in search_results[0]):
            logger.warning(f"Combined query returned no results, falling back to individual queries")
            # Try each query individually and combine results
            all_results = []
            seen_urls = set()
            
            for query in search_queries:
                try:
                    individual_results = await self.search_google(query, 5)  # Get fewer results per query
                    if individual_results and not (len(individual_results) == 1 and "error" in individual_results[0]):
                        for result in individual_results:
                            url = result.get("link", result.get("url", ""))
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                all_results.append(result)
                                if len(all_results) >= 15:  # Limit total results
                                    break
                    if len(all_results) >= 15:
                        break
                except Exception as e:
                    logger.debug(f"Error searching individual query '{query}': {e}")
                    continue
            
            if all_results:
                logger.debug(f"Fallback to individual queries returned {len(all_results)} results")
                search_results = all_results
            else:
                logger.warning(f"Failed to get search results from combined query and individual queries")
                return "Failed to get search results from Google."

        # Filter out huggingface.co results and keep max 8
        filtered_results = []
        for result in search_results:
            if "error" in result:
                continue
            url = result.get("link", result.get("url", ""))
            # Skip huggingface.co URLs
            if url and ("huggingface" in url.lower() or "gaia" in url.lower()):
                logger.debug(f"Filtering out huggingface or gaia URL: {url}")
                continue
            filtered_results.append(result)
            # Stop when we have 8 valid results
            if len(filtered_results) >= 8:
                break

        if not filtered_results:
            logger.warning(f"No valid search results found after filtering for combined query: {combined_query}")
            # Last resort: try individual queries if combined query failed
            if len(search_queries) > 1:
                logger.debug("Trying individual queries as last resort")
                all_results = []
                seen_urls = set()
                
                for query in search_queries:
                    try:
                        individual_results = await self.search_google(query, 5)
                        if individual_results and not (len(individual_results) == 1 and "error" in individual_results[0]):
                            for result in individual_results:
                                url = result.get("link", result.get("url", ""))
                                if url and url not in seen_urls and not ("huggingface" in url.lower() or "gaia" in url.lower()):
                                    seen_urls.add(url)
                                    all_results.append(result)
                                    if len(all_results) >= 8:
                                        break
                        if len(all_results) >= 8:
                            break
                    except Exception as e:
                        logger.debug(f"Error searching individual query '{query}': {e}")
                        continue
                
                if all_results:
                    filtered_results = all_results
                    logger.debug(f"Last resort individual queries returned {len(filtered_results)} results")
                else:
                    return "No valid search results found after filtering."
            else:
                return "No valid search results found after filtering."

        logger.debug(f"Using {len(filtered_results)} search results for combined query")

        # Step 3: Extract content from each webpage using JINA API
        async def extract_content_from_result(result):
            """Helper function to extract content from a single search result"""
            url = result.get("link", result.get("url", ""))
            if not url:
                return None

            logger.debug(f"Extracting content from URL: {url}")
            content = await self.crawler.crawl_jina(url)

            if content and not content.startswith("Error while extracting"):
                return {
                    "url": url,
                    "title": result.get("title", ""),
                    "content": content
                    if len(content) < 100000
                    else content[:100000] + f"...({len(content) - 100000} characters truncated)",
                    "search_query": combined_query,
                }
            return None

        # Use asyncio.gather for parallel content extraction
        extracted_contents = []

        async def task(result):
            async with asyncio.Semaphore(5):
                res = await extract_content_from_result(result)
                if res is not None:
                    extracted_contents.append(res)

        await asyncio.gather(*[task(result) for result in filtered_results])

        # URL deduplication
        seen_urls = set()
        unique_extracted_contents = []
        for content_item in extracted_contents:
            url = content_item["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                unique_extracted_contents.append(content_item)
            else:
                logger.debug(f"Skipping duplicate URL: {url}")

        extracted_contents = unique_extracted_contents

        if not extracted_contents:
            return "Failed to extract content from any of the search results."

        logger.debug(f"Successfully extracted content from {len(extracted_contents)} unique URLs")

        # Step 4: Use LLM to analyze the content and answer the question

        # If deep_info_page is enabled, summarize each page individually first
        if deep_info_page:
            logger.debug("deep_info_page enabled, summarizing each page individually")
            page_summaries = []

            async def summarize_single_page(content_item, index):
                """Helper function to summarize a single webpage"""
                # Truncate content to avoid exceeding model's max context length
                content = TokenUtils.truncate_text_by_token(
                    content_item["content"], limit=self.max_content_tokens
                )
                page_prompt = PROMPTS["DEEP_EXTRACTION"].format(
                    query=content_item["search_query"],
                    title=content_item["title"],
                    url=content_item["url"],
                    content=content,
                    question=question,
                )
                messages = [
                    {"role": "system", "content": PROMPTS["SP_SUMMARY"]},
                    {"role": "user", "content": page_prompt},
                ]
                logger.debug(f"Summarizing page {index + 1}: {content_item['url']}")
                try:
                    # page_summary = await self.llm.query_one(messages=messages)
                    # ---------------------------------------------------- 定制化Bob工具使用默认的qwen3-4b ----------------------------------------------------- #
                    page_summary = await self.llm.query_one(messages=messages, model="Qwen3-4B")
                    return {
                        "url": content_item["url"],
                        "title": content_item["title"],
                        "search_query": content_item["search_query"],
                        "summary": page_summary,
                        "index": index,  # Keep track of original order
                    }
                except Exception as e:
                    logger.error(f"Error summarizing page {index + 1} (URL: {content_item['url']}): {type(e).__name__}: {e}")
                    raise

            # Use asyncio.gather for parallel page summarization with concurrency control
            # Limit concurrent LLM requests to avoid connection errors
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent LLM requests
            
            async def task_with_semaphore(content_item, index):
                async with semaphore:
                    return await summarize_single_page(content_item, index)
            
            tasks = [task_with_semaphore(content_item, i) for i, content_item in enumerate(extracted_contents)]

            # Execute all summarization tasks concurrently
            temp_summaries = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and sort by original index
            page_summaries = []
            for summary_result in temp_summaries:
                if isinstance(summary_result, Exception):
                    # Error already logged in summarize_single_page, just skip
                    pass
                elif summary_result is not None:
                    page_summaries.append(summary_result)

            # Sort by original index to maintain order
            page_summaries.sort(key=lambda x: x["index"])
            page_summaries = [{k: v for k, v in summary.items() if k != "index"} for summary in page_summaries]

            # Prepare content from summaries for final analysis
            formatted_content = f"<search results for '{combined_query}'>\n"
            for i, summary_item in enumerate(page_summaries):
                formatted_content += f"[web_{i + 1}]\n"
                formatted_content += f"web title: {summary_item['title']}\n"
                formatted_content += f"web url: {summary_item['url']}\n"
                formatted_content += f"web summary: {summary_item['summary']}\n"
                if i < len(page_summaries) - 1:
                    formatted_content += "\n"
            formatted_content += "\n</search results end>"

        else:
            # Use the original formatting function to prepare content
            formatted_content = self._format_web_content(combined_query, extracted_contents)

        prompt = PROMPTS["SINGLE_DEEP_FINAL"].format(
            combined_query=combined_query,
            original_queries=", ".join(search_queries),
            formatted_content=formatted_content,
            question=question,
        )
        messages = [{"role": "system", "content": PROMPTS["SP_SINGLE_ANALYSIS"]}, {"role": "user", "content": prompt}]
        logger.debug("Sending prompt to LLM for final analysis")
        # final_answer = await self.llm.query_one(messages=messages)
        # ---------------------------------------------------- 定制化Bob工具使用默认的qwen3-4b ----------------------------------------------------- #
        final_answer = await self.llm.query_one(messages=messages, model="Qwen3-4B")

        # Add citations if add_citation is enabled
        if add_citation:
            logger.debug("add_citation enabled, appending citations")
            citations = "\n\n## References:\n"
            for i, content_item in enumerate(extracted_contents):
                citations += f"[{i + 1}] {content_item['title']} - {content_item['url']}\n"
            final_answer += citations

        return final_answer

    @register_tool
    async def parallel_search(self, search_queries: list[str], question: str) -> str:
        r"""Perform parallel search by using multiple Google search queries to search for different targets simultaneously and combining their snippets to answer a common question.

        Args:
            search_queries (List[str]): A list of search queries for different targets to be searched in parallel.
            question (str): The common question to ask about all search results.

        Returns:
            str: The LLM-generated answer to the question based on the search snippets from all queries.
        """
        logger.debug(f"Performing parallel search with {len(search_queries)} queries for question: {question}")

        all_search_results = []

        async def task(query):
            async with asyncio.Semaphore(5):
                res = await self.search_google(query, 5)
                all_search_results.append({"query": query, "results": res})

        await asyncio.gather(*[task(query) for query in search_queries])

        # URL去重 - 从所有搜索结果中移除重复的URL
        all_unique_results = []
        seen_urls = set()
        for query_data in all_search_results:
            unique_results = []
            for result in query_data["results"]:
                url = result.get("link", result.get("url", ""))
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
                elif url:
                    logger.debug(f"Skipping duplicate URL in parallel search: {url}")
            if unique_results:
                all_unique_results.append({"query": query_data["query"], "results": unique_results})
        all_search_results = all_unique_results

        if not all_search_results:
            return "Failed to get search results from any of the provided queries."

        # Step 2: Use LLM to analyze the snippets and answer the question

        # Step 3: Format the search results from all queries
        formatted_content = f"<search results for question: '{question}'>\n"

        answer_sections = []

        for query_data in all_search_results:
            query = query_data["query"]
            results = query_data["results"]

            # Format results for this query
            query_content = f"<search query: '{query}'>\n"

            for i, result in enumerate(results[:5]):  # Top 5 results
                url = result.get("link", result.get("url", ""))
                title = result.get("title", "")
                snippet = result.get("snippet", result.get("description", ""))

                query_content += f"<web{i + 1}>\n"
                query_content += f"url: {url}\n"
                query_content += f"title: {title}\n"
                query_content += f"snippet: {snippet}\n"
                query_content += f"</web{i + 1}>\n"

            query_content += "</search query end>\n\n"
            formatted_content += query_content

            # Get individual answer for this query
            individual_prompt = PROMPTS["PARALLEL_EXTRACTION"].format(
                query=query, query_content=query_content, question=question
            )
            messages = [
                {"role": "system", "content": PROMPTS["SP_PARALLEL"]},
                {"role": "user", "content": individual_prompt},
            ]
            logger.debug(f"Getting individual answer for query: {query}")
            # individual_answer = await self.llm.query_one(messages=messages)
            # ---------------------------------------------------- 定制化Bob工具使用默认的qwen3-4b ----------------------------------------------------- #
            individual_answer = await self.llm.query_one(messages=messages, model="Qwen3-4B")
            answer_sections.append({"query": query, "answer": individual_answer})

        formatted_content += "</search results end>"

        # Step 4: Format the final answer directly without LLM summarization
        final_answer = "<parallel_search_results>\n"
        final_answer += f"<question>{question}</question>\n\n"

        # Add query-specific answers
        final_answer += "<query_answers>\n"
        for i, section in enumerate(answer_sections, 1):
            final_answer += f"<answer_{i}>\n"
            final_answer += f"<search_query>{section['query']}</search_query>\n"
            final_answer += f"<analysis>{section['answer']}</analysis>\n"
            final_answer += f"</answer_{i}>\n\n"
        final_answer += "</query_answers>\n\n"

        # Add detailed search results to preserve all information
        final_answer += "<detailed_search_results>\n"
        final_answer += formatted_content
        final_answer += "</detailed_search_results>\n"
        final_answer += "</parallel_search_results>"

        return final_answer
