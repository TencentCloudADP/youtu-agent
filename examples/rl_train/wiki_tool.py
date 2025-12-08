# ruff: noqa: E501
import requests
import json
from utu.tools import AsyncBaseToolkit, register_tool
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from typing import Any, Callable, Optional, TypeVar
T = TypeVar("T")
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import ExitStack
from enum import Enum
from uuid import uuid4
from utu.config import ToolkitConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool mode enumeration."""
    ThreadMode = 1
    ProcessMode = 2


class TokenBucketLimiter:
    """Token bucket rate limiter using threading."""
    
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(rate_limit)
    
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        with self._lock:
            self.current_count += 1
    
    def release(self):
        """Release a token back to the bucket."""
        with self._lock:
            self.current_count -= 1
        self._semaphore.release()
    
    def get_current_count(self):
        """Get current number of acquired tokens."""
        with self._lock:
            return self.current_count


class SearchExecutionPool:
    """Execution pool for search operations with rate limiting."""
    
    def __init__(self, num_workers: int, enable_global_rate_limit: bool = True, rate_limit: int = 10):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.rate_limiter = TokenBucketLimiter(rate_limit) if enable_global_rate_limit else None
    
    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting.
        
        This method is synchronous but returns a Future-like object for compatibility.
        """
        if self.rate_limiter:
            self.rate_limiter.acquire()
            try:
                result = fn(*fn_args, **fn_kwargs)
            finally:
                self.rate_limiter.release()
            return result
        else:
            return fn(*fn_args, **fn_kwargs)
    
    def submit(self, fn: Callable[..., T], *fn_args, **fn_kwargs):
        """Submit function to thread pool for async execution."""
        def wrapped_func():
            if self.rate_limiter:
                with ExitStack() as stack:
                    # Acquire and ensure release
                    self.rate_limiter.acquire()
                    stack.callback(self.rate_limiter.release)
                    return fn(*fn_args, **fn_kwargs)
            else:
                return fn(*fn_args, **fn_kwargs)
        
        return self.executor.submit(wrapped_func)
    
    def shutdown(self):
        """Shutdown the thread pool."""
        self.executor.shutdown(wait=True)


def init_search_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return SearchExecutionPool(
            num_workers=num_workers,
            enable_global_rate_limit=enable_global_rate_limit,
            rate_limit=rate_limit
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")



class WikiToolkit(AsyncBaseToolkit):

    def __init__(self, config: ToolkitConfig = {}):
        super().__init__(config)
        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 90)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        # Retrieval service configuration
        # NOTE: you should change this to your own deployment
        self.retrieval_service_url = config.get("retrieval_service_url", "http://10.16.20.181:80/retrieve")
        self.topk = config.get("topk", 3)
        print(f"Initialized 🔍 WikiToolkit with config:{self.num_workers=}, {self.rate_limit=}, {self.timeout=}, {self.enable_global_rate_limit=}, {self.retrieval_service_url=}, {self.topk=}")
        logger.info(f"Initialized SearchTool with config: {config}")



    def execute_search(self, query_list: list, retrieval_service_url: str, topk: int, timeout: int=60):
        """Execute search operation using retrieval service.

        Args:
            # instance_id: Tool instance ID
            query_list: List of search queries
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        try:
            result_text_json = json.loads(result_text)
            result_text = result_text_json["result"]
        except:
            pass
        logger.debug(f"💻[wikipedia] searching {query_list} >>> total returned response:\n\n{result_text}")
        return result_text, metadata


    @register_tool
    async def search_wiki(self, query: list[str]) -> str:
        """Performs batched searches on wikipedia: supply an array 'query'; the tool retrieves the top 3 results for each query in one call.
        
        Args:
            query: Array of query strings. Include multiple complementary search queries in a single call.
        """
        
        '''
        # 古朴实现 没有任何管理措施
        result_text, metadata = perform_single_search_batch(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query,
                    topk=3,
                    concurrent_semaphore=None,  # Ray handles concurrency control
                    timeout=30,
                )
        result_text_str = json.loads(result_text)["result"]
        return result_text_str
        '''

        timeout = self.timeout
        if not query or not isinstance(query, list):
            error_msg = "Error: 'query' is missing, empty, or not a list."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return error_msg

        # Execute search using thread pool
        try:
            # Submit to thread pool and wait for result with timeout
            future = self.execution_pool.submit(
                self.execute_search, query,
                self.retrieval_service_url, self.topk, timeout
            )
            
            try:
                result_text, metadata = future.result(timeout=timeout)
                return result_text

            except FutureTimeoutError:
                error_msg = f"Search execution timed out after {timeout} seconds"
                logger.error(f"[SearchTool] {error_msg}")
                return error_msg
            
        except Exception as e:
            error_msg = f"Search execution failed: {e}"
            logger.error(f"[SearchTool] Execution failed: {e}")
            return error_msg

