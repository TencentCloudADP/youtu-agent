from typing import IO

from ..utils import FileUtils, get_logger
from .base_env import BasicEnv

logger = get_logger(__name__)


class E2BEnv(BasicEnv):
    """E2B Code Interpreter Environment.
    Ref: https://e2b.dev/docs
    Sample config: configs/agents/examples/e2b/e2b_python.yaml

    Variables:
        sandbox: AsyncSandbox instance

    Methods:
        files_read, files_write, files_edit_diff: File operations in the sandbox
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.request_timeout = config.get("request_timeout", 5)

    async def build(self):
        """Build the environment."""
        from e2b_code_interpreter import AsyncSandbox

        self.sandbox: AsyncSandbox = await AsyncSandbox.create(template="code-interpreter-v1", timeout=3600)
        logger.info(f"E2B sandbox created with id: {self.sandbox.sandbox_id}")

    async def cleanup(self):
        await self.sandbox.kill()

    def get_state(self) -> str:
        return ""

    async def files_read(self, path: str) -> str:
        try:
            return await self.sandbox.files.read(path, request_timeout=self.request_timeout)
        except Exception as e:
            logger.error(f"Error reading file {path} in e2b sandbox: {e}")
            return f"Error reading file {path}: {e}"

    async def files_write(self, path: str, data: str | bytes | IO) -> str:
        try:
            await self.sandbox.files.write(path, data, request_timeout=self.request_timeout)
            return f"Successfully wrote file: {path}"
        except Exception as e:
            logger.error(f"Error writing file {path} in e2b sandbox: {e}")
            return f"Error writing file {path}: {e}"

    async def files_edit_diff(self, path: str, diff: str) -> str:
        try:
            original_content = await self.sandbox.files.read(path, request_timeout=self.request_timeout)
            modified_content = FileUtils.apply_diff(original_content, diff)
            await self.sandbox.files.write(path, modified_content, request_timeout=self.request_timeout)
            return f"Successfully edited file: {path}"
        except Exception as e:
            logger.error(f"Error editing file {path} in e2b sandbox: {e}")
            return f"Error editing file {path}: {e}"
