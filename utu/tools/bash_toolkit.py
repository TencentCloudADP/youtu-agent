"""
https://github.com/pexpect/pexpect
@ii-agent/src/ii_agent/tools/bash_tool.py


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
import re
import subprocess
import os

from ..config import ToolkitConfig
from ..utils import get_logger
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


# Used to clean ANSI escape sequences
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


NSJAIL_PREFIX = """nsjail -q \
    -Mo --user 0 --group 99999 \
    -R /bin/ -R /lib/ -R /lib64/ \
    -R /usr/ -R /sbin/ -T /dev \
    -R /dev/urandom \
    -B {}:/tmp:rw \
    -R /etc/alternatives \
    -R /etc/matplotlibrc \
    -D /tmp \
    -E HOME=/tmp \
    -E LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    -E MPLCONFIGDIR=/tmp/.matplotlib \
    -E PATH=/usr/local/bin:/usr/bin:/bin --keep_caps -- /bin/bash
"""

class BashToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        # self.require_confirmation = self.config.config.get("require_confirmation", False)
        # self.command_filters = self.config.config.get("command_filters", [])
        self.timeout = self.config.config.get("timeout", 60)
        self.banned_command_strs = [
            "git init",
            "git commit",
            "git add",
        ]

        workspace_root = self.config.config.get("workspace_root", "/tmp/")
        self.setup_workspace(workspace_root)

    def setup_workspace(self, workspace_root: str):
        workspace_dir = pathlib.Path(workspace_root)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root = workspace_root
        os.makedirs('/tmp/.matplotlib', exist_ok=True)

    @register_tool
    async def run_bash(self, command: str) -> dict:
        """Execute a bash command in your workspace and return its output.

        Args:
            command: The command to execute
        """
        # 1) filter: change command before execution. E.g. used in SSH or Docker.
        # original_command = command
        # command = self.apply_filters(original_command)
        # if command != original_command:
        #     logger.info(f"Command filtered: {original_command} -> {command}")

        # 2) banned command check
        for banned_str in self.banned_command_strs:
            if banned_str in command:
                return f"Command not executed due to banned string in command: {banned_str} found in {command}."

        process = subprocess.Popen(
            NSJAIL_PREFIX.format(self.workspace_root),
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        stdout_data, stderr_data = process.communicate(input=command.encode("utf-8"))
        
        return_code = process.returncode
        stdout_result = stdout_data.decode("utf-8")
        stderr_result = stderr_data.decode("utf-8")
        
        stdout_result = ANSI_ESCAPE.sub("", stdout_result)
        stderr_result = ANSI_ESCAPE.sub("", stderr_result)
        
        if return_code == 0:
            return {
                "workdir": self.workspace_root,
                "success": True,
                "message": "Command executed successfully",
                "status": True,
                "stdout": stdout_result.strip(),
                "stderr": stderr_result.strip(),
            }
        
        return {
            "workdir": self.workspace_root,
            "success": False,
            "message": "Command execution failed",
            "status": False,
            "stdout": stdout_result.strip(),
            "stderr": stderr_result.strip(),
        }
