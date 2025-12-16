import pytest

from utu.config import ConfigLoader
from utu.tools import (
    CodesnipToolkit,
    GitHubToolkit,
    WikipediaSearchTool,
)


@pytest.fixture
def github_toolkit():
    return GitHubToolkit()


@pytest.fixture
async def wikipedia_toolkit():
    return WikipediaSearchTool(
        config={
            "user_agent": "uTu-agent",
            "language": "en",
            "content_type": "text",
            "extract_format": "WIKI",
        }
    )


async def test_get_repo_info(github_toolkit: GitHubToolkit):
    result = await github_toolkit.get_repo_info("https://github.com/github/github-mcp-server")
    assert result
    print(result)


async def test_wikipedia_search(wikipedia_toolkit: WikipediaSearchTool):
    result = await wikipedia_toolkit.wikipedia_search("Python_(programming_language)")
    print(result)


async def test_wikipedia_revisions(wikipedia_toolkit: WikipediaSearchTool):
    result = await wikipedia_toolkit.search_wikipedia_revisions("Penguin", 2022, 12)
    print(result)


@pytest.fixture
def codesnip_toolkit() -> CodesnipToolkit:
    config = ConfigLoader.load_toolkit_config("codesnip")
    return CodesnipToolkit(config=config)


async def test_run_code(codesnip_toolkit: CodesnipToolkit):
    result = await codesnip_toolkit.run_code("print('hello world')", "python")
    print(result)
