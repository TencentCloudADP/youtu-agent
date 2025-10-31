from agents import RunHooks

from ..config import AgentConfig
from .base_hooks import BaseRunHooks

__all__ = ["BaseRunHooks"]


def get_run_hooks(config: AgentConfig) -> RunHooks:
    # Currently, only BaseRunHooks is implemented.
    return BaseRunHooks()
