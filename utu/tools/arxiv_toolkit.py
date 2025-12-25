# ruff: noqa: E501
"""
https://info.arxiv.org/help/api/index.html
https://github.com/lukasschwab/arxiv.py
"""

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

from ..config import ToolkitConfig
from ..utils import get_logger
from .base import AsyncBaseToolkit

logger = get_logger(__name__)
if TYPE_CHECKING:
    import arxiv


class ArxivToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)

    def _get_search_results(
        self,
        query: str,
        paper_ids: list[str] | None = None,
        max_results: int | None = 5,
    ) -> Generator["arxiv.Result", None, None]:
        import arxiv

        paper_ids = paper_ids or []
        search_query = arxiv.Search(
            query=query,
            id_list=paper_ids,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,  # TODO: configurable, support advanced search
        )
        return arxiv.Client().results(search_query)

    async def search_papers(
        self,
        query: str,
        paper_ids: list[str] | None = None,
        max_results: int | None = 5,
    ) -> list[dict[str, str]]:
        r"""Searches for academic papers on arXiv using a query string and optional paper IDs.

        Args:
            query (str): The search query string.
            paper_ids (List[str], optional): A list of specific arXiv paper IDs to search for. (default None)
            max_results (int, optional): The maximum number of search results to return. (default 5)

        Tips:
            1. Use customized query for advanced search, including filtering and operators.
            2. filtering: ti for title, au for author, abs for abstract, all for all field match.
            3. operation: AND, OR, ANDNOT.
            4. data range: filter by paper's submittedDate, format YYYYMMDDTTTT. e.g. submittedDate:[202301010600 TO 202401010600]
            e.g. query="au:del_maestro AND ti:checkerboard" means search paper which has author Del Maestro and "checkerboard" keyword in title.
        """
        # https://info.arxiv.org/help/api/user-manual.html#query_details
        search_results = self._get_search_results(query, paper_ids, max_results)
        papers_data = []
        for paper in search_results:
            paper_info = {
                "title": paper.title,
                "published_date": paper.updated.date().isoformat(),
                "authors": [author.name for author in paper.authors],
                "entry_id": paper.entry_id,
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
            }
            # optional: extract text from paper with chunkr or arxiv_to_text
            # paper_info['paper_text'] = text
            papers_data.append(paper_info)
        return papers_data

    async def download_papers(
        self,
        query: str,
        paper_ids: list[str] | None = None,
        max_results: int | None = 5,
        output_dir: str | None = "./",
    ) -> str:
        r"""Downloads PDFs of academic papers from arXiv based on the provided query.

        Args:
            query (str): The search query string.
            paper_ids (List[str], optional): A list of specific arXiv paper IDs to download. (default: :obj:`None`)
            max_results (int, optional): The maximum number of search results to download. (default: :obj:`5`)
            output_dir (str, optional): The directory to save the downloaded PDFs. Defaults to the current directory.

        Returns:
            str: Status message indicating success or failure.
        """
        try:
            search_results = self._get_search_results(query, paper_ids, max_results)

            for paper in search_results:
                paper.download_pdf(dirpath=output_dir, filename=f"{paper.title}" + ".pdf")
            return "papers downloaded successfully"
        except Exception as e:  # pylint: disable=broad-except
            return f"An error occurred: {e}"

    async def get_tools_map(self) -> dict[str, Callable]:
        return {
            "search_papers": self.search_papers,
            "download_papers": self.download_papers,
        }
