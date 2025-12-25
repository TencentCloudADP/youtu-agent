"""
- [ ] support advanecd tools for file viewing & editing
    https://github.com/Intelligent-Internet/ii-agent/blob/main/src/ii_agent/tools/str_replace_tool.py
- [ ] context management (for long files)
"""

from ..config import ToolkitConfig
from ..utils import get_logger
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class FileEditToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)

        if self.env_mode == "local":
            from .local_env.file_edit import FileEditLocal

            self.file_editor = FileEditLocal(config)

    def setup_workspace(self, workspace_root: str):
        if self.env_mode != "local":
            logger.warning(f"FileEditToolkit should not setup workspace in env_mode {self.env_mode}!")
            return
        self.file_editor.setup_workspace(workspace_root)

    @register_tool
    async def edit_file(self, path: str, diff: str) -> str:
        r"""Edit a file by applying the provided diff.

        Args:
            file_name (str): The name of the file to edit.
            diff (str): (required) One or more SEARCH/REPLACE blocks following this exact format:
                ```
                <<<<<<< SEARCH
                [exact content to find]
                =======
                [new content to replace with]
                >>>>>>> REPLACE
                ```
        """
        if self.env_mode == "local":
            return await self.file_editor.edit_file(path, diff)
        else:
            assert self.e2b_sandbox is not None, "E2B sandbox is not set up!"
            return await self.e2b_env.files_edit_diff(path, diff)

    @register_tool
    async def write_file(self, path: str, file_text: str) -> str:
        """Write text content to a file.

        Args:
            path (str): The path of the file to write.
            file_text (str): The full text content to write.
        """
        if self.env_mode == "local":
            return await self.file_editor.write_file(path, file_text)
        else:
            assert self.e2b_sandbox is not None, "E2B sandbox is not set up!"
            return await self.e2b_env.files_write(path, file_text)

    @register_tool
    async def read_file(self, path: str) -> str:
        """Read and return the contents of a file.

        Args:
            path (str): The path of the file to read.
        """
        if self.env_mode == "local":
            return await self.file_editor.read_file(path)
        else:
            assert self.e2b_sandbox is not None, "E2B sandbox is not set up!"
            return await self.e2b_env.files_read(path)
