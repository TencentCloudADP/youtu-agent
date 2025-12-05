import asyncio
import base64
import contextlib
import glob
import io
import os
import queue
import re
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import ToolkitConfig
from .base import AsyncBaseToolkit, register_tool

# Used to clean ANSI escape sequences
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class ProcessPoolManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_workers: int = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_manager(max_workers)
            return cls._instance

    def _init_manager(self, max_workers: int):
        if max_workers is None:
            max_workers = 1

        self.max_workers = max_workers
        self.available_processes = queue.Queue(maxsize=max_workers)
        self.process_pools: dict[int, ProcessPoolExecutor] = {}
        self.lock = threading.Lock()

        for _ in range(max_workers):
            executor = ProcessPoolExecutor(max_workers=1)
            self.available_processes.put(executor)
            self.process_pools[id(executor)] = executor

    def acquire_process(self) -> ProcessPoolExecutor:
        try:
            return self.available_processes.get(timeout=300)
        except queue.Empty:
            raise RuntimeError("No available processes in pool")

    def release_process(self, executor: ProcessPoolExecutor):
        if id(executor) in self.process_pools:
            try:
                future = executor.submit(lambda: True)
                future.result(timeout=1)
                self.available_processes.put(executor)
            except:
                with self.lock:
                    if id(executor) in self.process_pools:
                        del self.process_pools[id(executor)]
                        new_executor = ProcessPoolExecutor(max_workers=1)
                        self.process_pools[id(new_executor)] = new_executor
                        self.available_processes.put(new_executor)

    def shutdown(self):
        with self.lock:
            for executor in self.process_pools.values():
                executor.shutdown(wait=False)
            self.process_pools.clear()
            while not self.available_processes.empty():
                try:
                    self.available_processes.get_nowait().shutdown(wait=False)
                except:
                    pass


_process_manager = None


def get_process_manager(max_workers: int = None) -> ProcessPoolManager:
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessPoolManager(max_workers)
    return _process_manager


def _execute_python_code_sync(code: str, workdir: str, max_memory_MB: int):
    """
    Synchronous execution of Python code.
    This function is intended to be run in a separate process.
    """
    original_dir = os.getcwd()
    try:
        # Clean up code format
        code_clean = code.strip()
        if code_clean.startswith("```python"):
            code_clean = code_clean.split("```python")[1].split("```")[0].strip()

        # memory limit
        memory_limit_code = f"""
import resource
try:
    memory_limit_bytes = {max_memory_MB} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
except (ValueError, resource.error):
    pass
"""
        code_clean = memory_limit_code + code_clean

        # Create and change to working directory
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)

        # Get file list before execution
        files_before = set(glob.glob("*"))

        # Create a new IPython shell instance
        from IPython.core.interactiveshell import InteractiveShell
        from traitlets.config.loader import Config

        InteractiveShell.clear_instance()

        config = Config()
        config.HistoryManager.enabled = False
        config.HistoryManager.hist_file = ":memory:"

        shell = InteractiveShell.instance(config=config)

        if hasattr(shell, "history_manager"):
            shell.history_manager.enabled = False

        output = io.StringIO()
        error_output = io.StringIO()

        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            shell.run_cell(code_clean)

            if plt.get_fignums():
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format="png")
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                plt.close()

                image_name = "output_image.png"
                counter = 1
                while os.path.exists(image_name):
                    image_name = f"output_image_{counter}.png"
                    counter += 1

                with open(image_name, "wb") as f:
                    f.write(base64.b64decode(img_base64))

        stdout_result = output.getvalue()
        stderr_result = error_output.getvalue()

        stdout_result = ANSI_ESCAPE.sub("", stdout_result)
        stderr_result = ANSI_ESCAPE.sub("", stderr_result)

        files_after = set(glob.glob("*"))
        new_files = list(files_after - files_before)
        new_files = [os.path.join(workdir, f) for f in new_files]

        try:
            shell.atexit_operations = lambda: None
            if hasattr(shell, "history_manager") and shell.history_manager:
                shell.history_manager.enabled = False
                shell.history_manager.end_session = lambda: None
            InteractiveShell.clear_instance()
        except Exception:
            pass

        return {
            "success": False
            if "Error" in stderr_result or ("Error" in stdout_result and "Traceback" in stdout_result)
            else True,
            "message": f"Code execution completed\nOutput:\n{stdout_result.strip()}"
            if stdout_result.strip()
            else "Code execution completed, no output",
            "status": True,
            "files": new_files,
            "error": stderr_result.strip() if stderr_result.strip() else "",
        }
    except Exception as e:
        return {
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
        max_workers = 32
        self.process_manager = get_process_manager(max_workers)

    async def get_tools_map(self) -> dict[str, callable]:
        return {
            "execute_python_code": self.execute_python_code,
        }

    @register_tool
    async def execute_python_code(
        self, code: str, workdir: str = "./run_workdir", timeout: int = 3, max_memory_MB: int = 512
    ) -> dict:
        """
        Executes Python code and returns the output.

        Args:
            code (str): The Python code to execute.
            workdir (str): The working directory for the execution. Defaults to "./run_workdir".
            timeout (int): The execution timeout in seconds. Defaults to 3.

        Returns:
            dict: A dictionary containing the execution results.
        """
        loop = asyncio.get_running_loop()
        starttime = time.time()
        executor = None

        try:
            executor = self.process_manager.acquire_process()
            res = await asyncio.wait_for(
                loop.run_in_executor(executor, _execute_python_code_sync, code, workdir, max_memory_MB),
                timeout=timeout,
            )
            res["time"] = time.time() - starttime
            return res
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
                "time": time.time() - starttime,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed: {e}",
                "stdout": "",
                "stderr": "",
                "status": False,
                "output": "",
                "files": [],
                "error": str(traceback.format_exc()),
                "time": time.time() - starttime,
            }
        finally:
            if executor is not None:
                self.process_manager.release_process(executor)
