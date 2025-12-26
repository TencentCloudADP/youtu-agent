import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from agents.function_schema import FuncSchema, function_schema

if TYPE_CHECKING:
    from e2b.sandbox.commands.command_handle import CommandExitException, CommandResult
    from e2b_code_interpreter.models import Execution


# ------------------------------------------------------------------------------
# e2b
class E2BUtils:
    @classmethod
    def execution_to_str(cls, execution: "Execution") -> str:
        """Convert e2b Execution to string.
        The official .to_json() is not good for Chinese output!"""
        from e2b_code_interpreter.models import serialize_results

        logs = execution.logs
        logs_data = {"stdout": logs.stdout, "stderr": logs.stderr}
        error = execution.error
        error_data = {"name": error.name, "value": error.value, "traceback": error.traceback} if error else None
        result = {
            "result": serialize_results(execution.results),
            "logs": logs_data,  # execution.logs.to_json(),
            "error": error_data,  # execution.error.to_json() if execution.error else None
        }
        return json.dumps(result, ensure_ascii=False)

    @classmethod
    def command_result_to_str(cls, command_result: "CommandResult") -> str:
        """Convert e2b CommandResult to string."""
        result = {
            "stdout": command_result.stdout,
            "stderr": command_result.stderr,
            "exit_code": command_result.exit_code,
            "error": command_result.error,
        }
        return json.dumps(result, ensure_ascii=False)

    @classmethod
    def command_exit_exception_to_str(cls, command_exception: "CommandExitException") -> str:
        result = {
            "stdout": command_exception.stdout,
            "stderr": command_exception.stderr,
            "exit_code": command_exception.exit_code,
            "error": command_exception.error,
        }
        return json.dumps(result, ensure_ascii=False)


# ------------------------------------------------------------------------------
# AsyncBaseToolkit utils
def register_tool(name: str = None):
    """Decorator to register a method as a tool.

    Usage:
        @register_tool  # uses method name
        @register_tool()  # uses method name
        @register_tool("custom_name")  # uses custom name

    Args:
        name (str, optional): The name of the tool. (Also support passing the function)
    """

    def decorator(func: Callable):
        if isinstance(name, str):
            tool_name = name
        else:
            tool_name = func.__name__
        func._is_tool = True
        func._tool_name = tool_name
        return func

    if callable(name):
        return decorator(name)
    return decorator


def get_tools_map(cls: type) -> dict[str, Callable]:
    """Get tools map from a class, without instance the class."""
    tools_map = {}
    # Iterate through all methods of the class and register @tool
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and getattr(attr, "_is_tool", False):
            tools_map[attr._tool_name] = attr
    return tools_map


def get_tools_schema(cls: type) -> dict[str, FuncSchema]:
    """Get tools schema from a class, without instance the class."""
    tools_map = {}
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and getattr(attr, "_is_tool", False):
            tools_map[attr._tool_name] = function_schema(attr)
    return tools_map


# ------------------------------------------------------------------------------
# misc
class ContentFilter:
    def __init__(self, banned_sites: list[str] = None):
        if banned_sites:
            self.RE_MATCHED_SITES = re.compile(r"^(" + "|".join(banned_sites) + r")")
        else:
            self.RE_MATCHED_SITES = None

    def filter_results(self, results: list[dict], limit: int, key: str = "link") -> list[dict]:
        # can also use search operator `-site:huggingface.co`
        # ret: {title, link, snippet, position, | sitelinks}
        res = []
        for result in results:
            if self.RE_MATCHED_SITES is None or not self.RE_MATCHED_SITES.match(result[key]):
                res.append(result)
            if len(res) >= limit:
                break
        return res
