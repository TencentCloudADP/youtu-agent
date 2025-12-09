"""
https://github.com/bytedance/SandboxFusion
https://bytedance.github.io/SandboxFusion/docs/docs/get-started
"""

import re

import requests

from ..config import ToolkitConfig
from ..utils import get_logger, oneline_object
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pttest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


class CodesnipToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        self.server_url = self.config.config.get("server_url")

    @staticmethod
    def _format_sandbox_result(result: dict) -> str:
        """Format SandboxFusion-style JSON into a human-readable message.

        This mirrors the post-processing logic used in `SandboxFusionTool.execute_code`
        on top of `_process_single_case` metadata:
        - Prefer stderr if present (show last 1–2 non-empty lines + a hint).
        - Otherwise return stdout, with a hint when stdout is empty.
        - Fallback to a generic error message when run status is not finished.
        """
        # Defensive access in case the schema or upstream service changes.
        data = result or {}

        # Preferred: assume the server already returns `_process_single_case`-style metadata.
        run_status = data.get("run_status")
        stdout = data.get("stdout")
        stderr = data.get("stderr")

        # Fallback: raw SandboxFusion response with nested `run_result`.
        if run_status is None and "run_result" in data:
            run_result = data.get("run_result") or {}
            run_status = run_result.get("status")
            stdout = run_result.get("stdout")
            stderr = run_result.get("stderr")

        if run_status == "Finished":
            std_error = stderr if stderr is not None else ""
            std_output = stdout if stdout is not None else ""

            if std_error:
                std_error = str(std_error)
                std_error_split = std_error.split("\n")
                std_error_split = [item.strip() for item in std_error_split if item.strip()]
                std_error = "\n".join(std_error_split[-2:])
                msg = std_error.strip()
                msg += "\nErrors occurred! Check your code."
            else:
                msg = str(std_output).strip()
                if not std_output:
                    msg += "\nEmpty stdout! You might forget to print the answer."
        else:
            msg = "Tool API failed! Please try again."

        return msg.strip()

    @staticmethod
    def _extract_code_from_markdown(code: str) -> str:
        """Extract code from markdown code blocks if present.

        Handles formats like:
        - ```python ... ```
        - ```py ... ```
        - ``` ... ```
        """
        # Pattern to match code blocks with optional language specifier
        pattern = r'```(?:python|py|Python)?\s*\n?(.*?)```'
        matches = re.findall(pattern, code, re.DOTALL)

        if matches:
            # Return the first matched code block content
            return matches[0].strip()

        # No code block found, return original code
        return code.strip()

    @register_tool("code_interpreter")
    async def code_interpreter(self, code: str) -> str:
        """A tool for executing code.
        """
        # Extract code from markdown code blocks if present
        code = self._extract_code_from_markdown(code)

        # Only expose `code` in the tool schema; internally we fix language to python.
        language = "python"

        cfg = self.config.config or {}
        # Align with VERL: use a single timeout value for both compile and run if not explicitly provided.
        default_timeout = cfg.get("default_timeout", 30)
        compile_timeout = cfg.get("compile_timeout", default_timeout)
        run_timeout = cfg.get("run_timeout", default_timeout)
        memory_limit_mb = cfg.get("memory_limit_mb", 1024)

        # Match the payload structure used by VERL's `call_sandbox_api`.
        payload = {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": None,
            "memory_limit_MB": memory_limit_mb,
            "language": language,
            "files": {},
            "fetch_files": [],
        }

        response = requests.post(
            f"{self.server_url}/run_code",
            json=payload,
            timeout=compile_timeout + run_timeout + 10,
        )
        result = response.json()
        logger.info(f"[tool] code_interpreter ```{oneline_object(payload)}``` got result: {oneline_object(result)}")
        # Post-process the raw JSON result into a concise text message,
        # consistent with the behavior of `SandboxFusionTool.execute_code`.
        return self._format_sandbox_result(result)
