"""Document toolkit for parsing documents and support Q&A.

Support backends:

- Chunkr: <https://github.com/lumina-ai-inc/chunkr>
- pymupdf: <https://github.com/pymupdf/PyMuPDF>
- unstructured: <https://github.com/Unstructured-IO/unstructured>

- [ ] unify the filepath cache logic (also suppoort audio_toolkit, image_toolkit)
"""

import json
import pathlib

from ..config import ToolkitConfig
from ..utils import CACHE_DIR, FileUtils, SimplifiedAsyncOpenAI, get_logger
from .base import TOOL_PROMPTS, AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class DocumentToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        """Initialize the DocumentToolkit, with configed parser and llm."""
        super().__init__(config)
        if self.config.config.get("parser") == "chunkr":
            from .documents.chunkr_parser import ChunkrParser

            self.parser = ChunkrParser(self.config.config)
        elif self.config.config.get("parser") == "pymupdf":
            from .documents.pdf_parser import PDFParser

            self.parser = PDFParser(self.config.config)
        elif self.config.config.get("parser") == "unstructured":
            from .documents.unstructured_parser import UnstructuredParser

            self.parser = UnstructuredParser(self.config.config)
        else:
            raise ValueError(f"Unsupported parser: {self.config.config.get('parser')}")
        self.text_limit = self.config.config.get("text_limit", 100_000)
        self.llm = SimplifiedAsyncOpenAI(**self.config.config_llm.model_provider.model_dump())
        self.md5_to_path = {}

    async def parse_document(self, md5: str) -> str:
        logger.info(f"[tool] parse_document: {self.md5_to_path[md5]}")
        return await self.parser.parse(self.md5_to_path[md5])

    def handle_path(self, path_or_url: str) -> str:
        if FileUtils.is_web_url(path_or_url):
            md5 = FileUtils.get_file_md5(path_or_url)
            logger.info(f"md5 for {path_or_url}: {md5}")
            # download document to data/_document, with md5
            fn = CACHE_DIR / "documents" / f"{md5}{FileUtils.get_file_ext(path_or_url)}"
            fn.parent.mkdir(parents=True, exist_ok=True)
            if not fn.exists():
                logger.info(f"Downloaded document file to {path_or_url}")
                FileUtils.download_file(url=path_or_url, save_path=fn)
            self.md5_to_path[md5] = fn  # record md5 to map
            return md5

        # normalize local path (expand ~, make absolute) before hashing
        local_path = pathlib.Path(path_or_url).expanduser()
        if not local_path.is_absolute():
            base_dir = self.config.config.get("workspace_root") or pathlib.Path.home()
            try:
                base_dir_path = pathlib.Path(base_dir).expanduser()
            except TypeError:
                base_dir_path = pathlib.Path.home()
            local_path = base_dir_path / local_path
        local_path = local_path.resolve()

        md5 = FileUtils.get_file_md5(str(local_path))
        logger.info(f"md5 for {local_path}: {md5}")
        self.md5_to_path[md5] = local_path
        return md5

    @register_tool
    async def document_parse(self, document_path: str, chunk_size: int = None, chunk_id: int = None) -> str:
        """Parse document and return the processed text.
        - Supported file types: pdf, docx, pptx, xlsx, xls, ppt, doc
        - If the document is too large, it will be truncated to the first chunk_size characters.
        - If pass chunk_id, it will return the chunk text begin with chunk_id * chunk_size.

        Args:
            document_path (str): Local path or URL to a document.
            chunk_size (int, optional): Number of characters to process at once. Defaults to 10_000.
            chunk_id (int, optional): Chunk ID to start from. Defaults to 0.
        """
        md5 = self.handle_path(document_path)
        document_markdown = await self.parse_document(md5)

        meta = {
            "path": self.md5_to_path[md5],
            "total_chars": len(document_markdown),
        }
        chunk_size = chunk_size or 10_000
        chunk_id = chunk_id or 0
        if meta["total_chars"] > chunk_size:
            meta["is_chunked"] = True
            meta["chunk_size"] = chunk_size
            meta["chunk_total"] = (meta["total_chars"] + chunk_size - 1) // chunk_size
            meta["chunk_id"] = chunk_id
            meta["content"] = document_markdown[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
        else:
            meta["is_chunked"] = False
            meta["content"] = document_markdown
        return json.dumps(meta, ensure_ascii=False)

    @register_tool
    async def document_qa(self, document_path: str, question: str | None = None) -> str:
        """Get file content summary or answer questions about attached document.

        Supported file types: pdf, docx, pptx, xlsx, xls, ppt, doc

        Args:
            document_path (str): Local path or URL to a document.
            question (str, optional): The question to answer. If not provided, return a summary of the document.
        """
        
        md5 = self.handle_path(document_path)
        document_markdown = await self.parse_document(md5)
        if len(document_markdown) > self.text_limit:
            document_markdown = document_markdown[: self.text_limit] + "\n..."
        messages = [
            {"role": "system", "content": TOOL_PROMPTS["document_sp"]},
            {"role": "user", "content": document_markdown},
        ]
        if question:
            messages.append({"role": "user", "content": TOOL_PROMPTS["document_qa"].format(question=question)})
        else:
            messages.append({"role": "user", "content": TOOL_PROMPTS["document_summary"]})
        output = await self.llm.query_one(messages=messages, **self.config.config_llm.model_params.model_dump())
        if not question:
            output = (
                f"You did not provide a particular question, so here is a detailed caption for the document: {output}"
            )
        return output
