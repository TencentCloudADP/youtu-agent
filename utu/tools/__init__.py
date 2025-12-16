from ..config import ConfigLoader
from .arxiv_toolkit import ArxivToolkit
from .audio_toolkit import AudioToolkit
from .base import AsyncBaseToolkit as AsyncBaseToolkit
from .bash_toolkit import BashToolkit
from .codesnip_toolkit import CodesnipToolkit
from .document_toolkit import DocumentToolkit
from .file_edit_toolkit import FileEditToolkit
from .github_toolkit import GitHubToolkit
from .image_toolkit import ImageToolkit
from .memory_toolkit import SimpleMemoryToolkit
from .python_executor_toolkit import PythonExecutorToolkit
from .search_toolkit import SearchToolkit
from .serper_toolkit import SerperToolkit
from .tabular_data_toolkit import TabularDataToolkit
from .user_interaction_toolkit import UserInteractionToolkit
from .utils import get_tools_map as get_tools_map, get_tools_schema as get_tools_schema, register_tool as register_tool
from .video_toolkit import VideoToolkit
from .wikipedia_toolkit import WikipediaSearchTool

TOOLKIT_MAP: dict[str, type[AsyncBaseToolkit]] = {
    "search": SearchToolkit,
    "document": DocumentToolkit,
    "image": ImageToolkit,
    "file_edit": FileEditToolkit,
    "github": GitHubToolkit,
    "arxiv": ArxivToolkit,
    "wikipedia": WikipediaSearchTool,
    "codesnip": CodesnipToolkit,
    "bash": BashToolkit,
    "python_executor": PythonExecutorToolkit,
    "video": VideoToolkit,
    "audio": AudioToolkit,
    "serper": SerperToolkit,
    "tabular": TabularDataToolkit,
    "memory_simple": SimpleMemoryToolkit,
    "user_interaction": UserInteractionToolkit,
}


def get_toolkits_map(names: list[str] | None = None) -> dict[str, AsyncBaseToolkit]:
    """Get all the toolkits specified by names

    WARNING: not used for now
    """
    toolkits = {}
    if names is None:
        names = list(TOOLKIT_MAP.keys())
    else:
        assert all(name in TOOLKIT_MAP for name in names), f"Error config tools: {names}"
    for name in names:
        config = ConfigLoader.load_toolkit_config(name)
        toolkits[name] = TOOLKIT_MAP[name](config=config)
    return toolkits
