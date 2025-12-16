"""
--- https://www.anthropic.com/engineering/swe-bench-sonnet ---
Run commands in a bash shell\n
* When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.\n
* You don't have access to the internet via this tool.\n
* You do have access to a mirror of common linux and python packages via apt and pip.\n
* State is persistent across command calls and discussions with the user.\n
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n
* Please avoid commands that may produce a very large amount of output.\n
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background."
"""

import pathlib

from ..config import ToolkitConfig
from ..utils import get_logger
from .base import AsyncBaseToolkit, register_tool
from .utils import E2BUtils

logger = get_logger(__name__)


class BashToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        self.timeout = self.config.config.get("timeout", 60)

        if self.env_mode == "local":
            from .local_env.bash_pexpect import PexpectBash

            self.bash_runner = PexpectBash(timeout=self.timeout)
            self.setup_workspace(self.config.config.get("workspace_root", "/tmp/"))

    def setup_workspace(self, workspace_root: str):
        if self.env_mode != "local":
            logger.warning(f"BashToolkit should not setup workspace in env_mode {self.env_mode}!")
            return
        workspace_dir = pathlib.Path(workspace_root)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root = workspace_root
        self.bash_runner.run(f"cd {workspace_root}")

    @register_tool
    async def run_bash(self, command: str) -> str:
        """Execute a bash command in your workspace and return its output.

        Args:
            command: The command to execute
        """
        if self.env_mode == "local":
            return self.bash_runner.run(command)
        else:
            assert self.e2b_sandbox is not None, "E2B sandbox is not set up!"
            from e2b.sandbox.commands.command_handle import CommandExitException

            try:
                result = await self.e2b_sandbox.commands.run(command, timeout=self.timeout)
                return E2BUtils.command_result_to_str(result)
            except CommandExitException as e:
                return E2BUtils.command_exit_exception_to_str(e)
