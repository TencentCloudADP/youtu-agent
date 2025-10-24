"""
- [ ] polish _execute_python_code_sync
"""

import asyncio
import base64
import contextlib
import glob
import io
import os
import pathlib
import re
import traceback
import uuid
from datetime import datetime
import subprocess

from ..config import ToolkitConfig
from .base import AsyncBaseToolkit, register_tool


# Used to clean ANSI escape sequences
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


NSJAIL_PREFIX = """nsjail -q \
    -Mo --user 0 --group 99999 \
    -R /bin/ -R /lib/ -R /lib64/ \
    -R /usr/ -R /sbin/ -T /dev \
    -R /dev/urandom \
    -R /etc/alternatives  \
    -B {}:/tmp:rw \
    -D /tmp \
    -E LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    -E HOME=/tmp \
    -E MPLCONFIGDIR=/tmp/.matplotlib \
    -E PATH=/usr/local/bin:/usr/bin:/bin --keep_caps -- /usr/bin/mkdir -p /tmp/.matplotlib && /usr/bin/python3 -
"""


def _execute_python_code_sync(code: str, workdir: str):
    """
    Synchronous execution of Python code.
    This function is intended to be run in a separate thread.
    """
    original_dir = os.getcwd()
    try:
        # Clean up code format
        code_clean = code.strip()
        if code_clean.startswith("```python"):
            code_clean = code_clean.split("```python")[1].split("```")[0].strip()

        # Create and change to working directory
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)

        # Get file list before execution
        files_before = set(glob.glob("*"))

        # run in nsjail
        process = subprocess.Popen(NSJAIL_PREFIX.format(workdir), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        stdout_data, stderr_data = process.communicate(input=code_clean.encode("utf-8"), timeout=5)

        stdout_result = stdout_data.decode("utf-8")
        stderr_result = stderr_data.decode("utf-8")
        
        # print("stdout_result: ", stdout_result)
        # print("stderr_result: ", stderr_result)

        stdout_result = ANSI_ESCAPE.sub("", stdout_result)
        stderr_result = ANSI_ESCAPE.sub("", stderr_result)

        files_after = set(glob.glob("*"))
        new_files = list(files_after - files_before)
        new_files = [os.path.join(workdir, f) for f in new_files]

        success = True
        if "Error" in stderr_result or ("Error" in stdout_result and "Traceback" in stdout_result):
            success = False
        message = "Code execution completed, no output"
        if stdout_result.strip():
            message = f"Code execution completed\nOutput:\n{stdout_result.strip()}"

        return {
            "workdir": workdir,
            "success": success,
            "message": message,
            "status": True,
            "files": new_files,
            "error": stderr_result.strip(),
        }
    except Exception as e:  # pylint: disable=broad-except
        return {
            "workdir": workdir,
            "success": False,
            "message": f"Code execution failed, error message:\n{str(e)},\nTraceback:{traceback.format_exc()}",
            "status": False,
            "files": [],
            "error": str(e),
        }
    finally:
        os.chdir(original_dir)


class PythonExecutorToolkit(AsyncBaseToolkit):
    """
    A tool for executing Python code in a sandboxed environment.
    """

    def __init__(self, config: ToolkitConfig | dict | None = None):
        super().__init__(config)

        workspace_root = self.config.config.get("workspace_root", None)
        if workspace_root is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            workspace_root = f"/tmp/utu/python_executor/{timestamp}_{unique_id}"
        self.setup_workspace(workspace_root)

    def setup_workspace(self, workspace_root: str):
        workspace_dir = pathlib.Path(workspace_root)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root = workspace_root

    @register_tool
    async def execute_python_code(self, code: str, timeout: int = 30) -> dict:
        """
        Executes Python code and returns the output.

        Args:
            code (str): The Python code to execute.
            timeout (int): The execution timeout in seconds. Defaults to 30.

        Returns:
            dict: A dictionary containing the execution results.
        """
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,  # Use the default thread pool executor
                    _execute_python_code_sync,
                    code,
                    str(self.workspace_root),
                ),
                timeout=timeout,
            )
        except TimeoutError:
            return {
                "success": False,
                "message": f"Code execution timed out ({timeout} seconds)",
                "stdout": "",
                "stderr": "",
                "status": False,
                "output": "",
                "files": [],
                "error": f"Code execution timed out ({timeout} seconds)",
            }
