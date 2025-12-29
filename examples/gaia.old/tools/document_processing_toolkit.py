# ruff: noqa
"""
- [ ] support filenames like .MOV
- [ ] support .pdb
- [x] support .xlsx
- [x] add image model
- [x] add audio model
- [ ] support video
"""

import os
import asyncio
import subprocess
import xmltodict
import nest_asyncio
import requests
import validators
import json
import pathlib
import fitz

from docx2markdown._docx_to_markdown import docx_to_markdown
from typing import List, Optional, Tuple, Callable
from urllib.parse import urlparse
from unstructured.partition.auto import partition

from examples.gaia.tools.excel_toolkit import ExcelToolkit
from examples.gaia.tools.image_analysis_toolkit import ImageAnalysisToolkit
from examples.gaia.tools.audio_analysis_toolkit import AudioAnalysisToolkit

from utu.config import ToolkitConfig, ConfigLoader
from utu.tools import AsyncBaseToolkit
from utu.tools.search.jina_crawl import JinaCrawl
from utu.utils import get_logger, SimplifiedAsyncOpenAI, FileUtils

logger = get_logger(__name__)
PROMPTS = FileUtils.load_prompts(pathlib.Path(__file__).parent / "document_processing_prompts.yaml")

nest_asyncio.apply()


class DocumentProcessingToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)

        self.image_tool = ImageAnalysisToolkit(ConfigLoader.load_toolkit_config("image"))
        self.audio_tool = AudioAnalysisToolkit(ConfigLoader.load_toolkit_config("audio"))
        self.excel_tool = ExcelToolkit()
        self.crawler = JinaCrawl()

        # llm for web_qa
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        }

        self.cache_dir = "workspace/"
        os.makedirs(os.path.join(self.cache_dir, "docx_cache"), exist_ok=True)

    async def extract_document_content(self, document_path: str, query: Optional[str] = None) -> str:
        r"""Extract the content of a given document and return the processed text.
        It may filter out some information, resulting in inaccurate content.

        Args:
            document_path (str): The path of the document to be processed, either a local path or a URL. It can process image, audio files, zip files and webpages, etc.
            query (str): The query to be used for retrieving the content. If the content is too long, the query will be used to identify which part contains the relevant information (like RAG). The query should be consistent with the current task.
        """
        logger.debug(f"Calling extract_document_content function with document_path=`{document_path}`")

        parsed_url = urlparse(document_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])

        if any(document_path.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            res = await self.image_tool.ask_question_about_image(
                document_path, "Please make a detailed caption about the image."
            )
            return res

        if any(document_path.endswith(ext) for ext in [".mp3", ".wav"]):
            res = await self.audio_tool.ask_question_about_audio(
                document_path, "Please transcribe the audio content to text."
            )
            return res

        if any(document_path.endswith(ext) for ext in ["txt"]):
            if is_url:
                document_path = self._download_file(document_path)
            with open(document_path, "r", encoding="utf-8") as f:
                content = f.read()
            f.close()
            res = await self._post_process_result(content, query)
            return res

        if any(document_path.endswith(ext) for ext in ["xls", "xlsx"]):
            if is_url:
                document_path = self._download_file(document_path)
            res = await self.excel_tool.extract_excel_content(document_path)
            return res

        if any(document_path.endswith(ext) for ext in ["zip"]):
            extracted_files = self._unzip_file(document_path)
            return f"The extracted files are: {extracted_files}"

        if any(document_path.endswith(ext) for ext in ["json", "jsonl", "jsonld"]):
            with open(document_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            f.close()
            return content

        if any(document_path.endswith(ext) for ext in ["py"]):
            with open(document_path, "r", encoding="utf-8") as f:
                content = f.read()
            f.close()
            return content

        if any(document_path.endswith(ext) for ext in ["xml"]):
            data = None
            with open(document_path, "r", encoding="utf-8") as f:
                content = f.read()
            f.close()

            data = xmltodict.parse(content)
            logger.debug(f"The extracted xml data is: {data}")
            return data

        if self._is_webpage(document_path):
            extracted_text = await self.crawler.crawl(document_path)
            result_filtered = await self._post_process_result(extracted_text, query)
            return result_filtered
        else:
            # judge if url
            parsed_url = urlparse(document_path)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if not is_url:
                if not os.path.exists(document_path):
                    return f"Document not found at path: {document_path}."

            # if is docx file, use docx2markdown to convert it
            if document_path.endswith(".docx"):
                if is_url:
                    tmp_path = self._download_file(document_path)
                else:
                    tmp_path = document_path

                file_name = os.path.basename(tmp_path)
                os.makedirs(os.path.join(self.cache_dir, "docx_cache"), exist_ok=True)
                # md_file_path = f"{file_name}.md"
                md_file_path = os.path.join(self.cache_dir, "docx_cache", f"{file_name}.md")
                if os.path.exists(md_file_path):
                    logger.debug(f"Using cached md file: {md_file_path}")
                else:
                    logger.debug(f"Converting docx to markdown: {tmp_path} -> {md_file_path}")
                    docx_to_markdown(tmp_path, md_file_path)

                # load content of md file
                with open(md_file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                f.close()
                return extracted_text

            if document_path.endswith(".pptx"):
                # use unstructured to extract text from pptx
                extracted_text = partition(document_path)
                extracted_text = [item.text for item in extracted_text]
                return extracted_text

            if document_path.endswith(".pdf"):
                # try using fitz (PyMuPDF) to extract text from pdf
                if is_url:
                    tmp_path = self._download_file(document_path)
                    document_path = tmp_path
                doc = fitz.open(document_path)
                extracted_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    extracted_text += f"## Page {page_num + 1}\n\n"
                    if text.strip():
                        extracted_text += text.strip() + "\n\n"
                doc.close()

                result_filtered = await self._post_process_result(extracted_text, query)
                return result_filtered

            # use unstructured to extract text from file
            extracted_text = partition(document_path)
            extracted_text = [item.text for item in extracted_text]
            return extracted_text

    async def extract_web_content(self, url: str, query: str) -> str:
        r"""Extract and analyze webpage content given a webpage URL and a query, and return the processed text based on the query.

        Args:
            url (str): The URL of the webpage to be processed.
            query (str): A specific query to ask about the webpage content. The content will be processed through LLM Q&A based on this query.

        Returns:
            str: The processed text based on the query.
        """
        logger.debug(f"Calling extract_web_content function with url=`{url}`, query=`{query}`")
        enable_qa = True

        # Check if it's a valid webpage
        if not self._is_webpage(url):
            return f"The provided URL is not a valid webpage: {url}"

        # Extract webpage content
        extracted_text = await self.crawler.crawl(url)
        if not extracted_text or extracted_text.startswith("Error while extracting"):
            return f"Failed to extract content from webpage: {url}"
        logger.debug(f"Successfully extracted content from webpage: {url}")

        # Post-process the result with query
        result_filtered = await self._post_process_result_pro(extracted_text, query)

        # Apply Q&A if enabled
        if enable_qa:
            logger.debug(f"enable_qa is True, processing with Q&A for query: {query}")
            qa_result = await self._process_with_qa(result_filtered, query)
            return qa_result
        else:
            return result_filtered

    async def _process_with_qa(self, content: str, query: str) -> str:
        r"""Process the extracted content with Q&A using LLM."""
        prompt = PROMPTS["FINAL_QA"].format(content=content, query=query)
        logger.debug(f"Processing Q&A with content length: {len(content)}")
        return await self.llm.query_one(messages=prompt)

    async def _post_process_result(self, result: str, query: Optional[str]) -> str:
        r"""Identify whether the result is too long. If so, split it into multiple parts, and leverage a model to identify which part contains the relevant information."""

        async def _identify_relevant_part_async(part_idx: int, part: str, query: str) -> Tuple[bool, int, str]:
            logger.debug(f"Doc understanding with length {len(part)}, query: {query}")
            prompt = PROMPTS["POSTPROCESS_QA_RELEVANCE"].format(part=part, query=query)
            response = await self.llm.query_one(messages=prompt)
            flag = "true" in response.lower()
            return flag, part_idx, part

        max_length = 100000
        split_length = 40000
        max_truncate_length = 500000

        # Truncate result if it exceeds max_truncate_length
        if len(result) > max_truncate_length:
            result = result[:max_truncate_length]
            logger.debug(f"Result truncated to {max_truncate_length} characters")

        if len(result) > max_length:
            # Handle None query
            if query is None:
                return result[:max_length]

            # split the result into multiple parts
            logger.debug(f"The original result is too long. Splitting it into multiple parts. query: {query}")
            parts = [result[i : i + split_length] for i in range(0, len(result), split_length)]
            result_cache = {}

            # Use asyncio.gather for parallel processing
            tasks = [_identify_relevant_part_async(part_idx, part, query) for part_idx, part in enumerate(parts)]

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result_item in results:
                    if isinstance(result_item, Exception):
                        logger.error(f"Error in parallel processing: {result_item}")
                        continue
                    is_relevant, part_idx, part = result_item
                    if is_relevant:
                        result_cache[part_idx] = part
            except Exception as e:
                logger.error(f"Error in asyncio.gather: {e}")
                return result[:max_length]

            # re-assemble the parts according to the part_idx
            result_filtered = ""
            for part_idx in sorted(result_cache.keys()):
                result_filtered += result_cache[part_idx]
                result_filtered += "..."

            result_filtered += "(The above is the re-assembled result of the document, because the original document is too long. If empty, it means no relevant information found.)"
            if len(result_filtered) > max_length:
                result_filtered = result_filtered[:max_length]  # TODO: Refine it to be more accurate
            logger.debug(f"split context length: {len(result_filtered)}")
            return result_filtered

        else:
            return result

    async def _post_process_result_pro(self, result: str, query: str) -> str:
        r"""Advanced post-processing that performs progressive Q&A on document chunks.
        Unlike _post_process_result, this method not only identifies relevant chunks but also
        progressively builds answers by calling LLM for Q&A on each relevant chunk.
        """

        async def _identify_relevant_part(part_idx: int, part: str, query: str) -> Tuple[bool, int, str]:
            logger.debug(f"Doc understanding with length {len(part)}, query: {query}")

            prompt = PROMPTS["POSTPROCESS_QA_RELEVANCE"].format(part=part, query=query)

            response = await self.llm.query_one(messages=prompt)
            flag = "true" in response.lower()
            return flag, part_idx, part

        async def _progressive_qa(part: str, query: str, previous_answer: str) -> str:
            """Perform Q&A on a document part, considering previous answers"""
            if previous_answer:
                prompt = PROMPTS["POSTPROCESS_QA_PROGRESSIVE"].format(
                    part=part, query=query, previous_answer=previous_answer
                )
            else:
                prompt = PROMPTS["POSTPROCESS_QA_FIRST"].format(part=part, query=query)

            return await self.llm.query_one(messages=prompt)

        max_length = 100000
        max_cut_length = 200000
        split_length = 40000

        if len(result) > max_length:
            # Split the result into multiple parts
            logger.debug(
                f"The original result is too long. Splitting it into multiple parts for progressive Q&A. query: {query}"
            )
            if len(result) > max_cut_length:
                logger.debug(f"Result length {len(result)} exceeds max_cut_length {max_cut_length}, truncating.")
                result = result[:max_cut_length]
            parts = [result[i : i + split_length] for i in range(0, len(result), split_length)]
            relevant_parts = {}

            # First, identify all relevant parts using async
            tasks = [_identify_relevant_part(part_idx, part, query) for part_idx, part in enumerate(parts)]

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result_item in results:
                    if isinstance(result_item, Exception):
                        logger.error(f"Error in parallel processing: {result_item}")
                        continue
                    is_relevant, part_idx, part = result_item
                    if is_relevant:
                        relevant_parts[part_idx] = part
            except Exception as e:
                logger.error(f"Error in asyncio.gather: {e}")
                return "(Error occurred while processing document parts.)"

            if not relevant_parts:
                return "(No relevant information found in the document.)"

            # Progressive Q&A on relevant parts in order
            logger.debug(f"Found {len(relevant_parts)} relevant parts, performing progressive Q&A")
            progressive_answer = ""

            for part_idx in sorted(relevant_parts.keys()):
                part_content = relevant_parts[part_idx]
                logger.debug(f"Processing part {part_idx + 1} with progressive Q&A")

                try:
                    progressive_answer = await _progressive_qa(part_content, query, progressive_answer)
                    logger.debug(f"Updated answer after processing part {part_idx + 1}")
                except Exception as e:
                    logger.error(f"Error in progressive Q&A for part {part_idx + 1}: {e}")
                    # If Q&A fails, skip this part and continue with next part
                    logger.warning(f"Skipping part {part_idx + 1} due to Q&A error")
                    continue

            final_result = progressive_answer

            logger.debug(f"Progressive Q&A completed, final answer length: {len(final_result)}")
            return final_result

        else:
            # For shorter documents, perform single Q&A
            logger.debug("Document is short enough, performing single Q&A")
            prompt = PROMPTS["POSTPROCESS_QA_SHORT"].format(result=result, query=query)
            return await self.llm.query_one(messages=prompt)

    def _is_webpage(self, url: str) -> bool:
        r"""Judge whether the given URL is a webpage."""
        return validators.url(url) is True

    def _download_file(self, url: str):
        r"""Download a file from a URL and save it to the cache directory."""
        try:
            file_name = url.split("/")[-1]
            file_path = os.path.join(self.cache_dir, file_name)
            if os.path.exists(file_path):
                logger.debug(f"File already exists in cache: {file_path}")
            else:
                logger.debug(f"Downloading file from {url} to {file_path}")

                response = requests.get(url, stream=True, headers=self.headers)
                response.raise_for_status()

                with open(file_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

            return file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")

    def _unzip_file(self, zip_path: str) -> List[str]:
        if not zip_path.endswith(".zip"):
            raise ValueError("Only .zip files are supported")

        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_path = os.path.join(self.cache_dir, zip_name)
        os.makedirs(extract_path, exist_ok=True)

        try:
            subprocess.run(["unzip", "-o", zip_path, "-d", extract_path], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to unzip file: {e}")

        extracted_files = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                extracted_files.append(os.path.join(root, file))

        return extracted_files

    async def get_tools_map(self) -> dict[str, Callable]:
        return {
            "extract_document_content": self.extract_document_content,
            "extract_web_content": self.extract_web_content,
        }
