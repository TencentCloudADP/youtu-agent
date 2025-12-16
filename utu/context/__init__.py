from ..config import AgentConfig
from .base_context_manager import BaseContextManager, DummyContextManager
from .env_context_manager import EnvContextManager


def build_context_manager(config: AgentConfig):
    if (not config.context_manager) or (not config.context_manager.name):
        return DummyContextManager()
    match config.context_manager.name:
        case "dummy":
            return DummyContextManager(config.context_manager.config)
        case "env":
            return EnvContextManager(config.context_manager.config)
        case _:
            raise ValueError(f"Unknown context manager: {config.context_manager.name}")


__all__ = ["build_context_manager", "DummyContextManager", "EnvContextManager", "BaseContextManager"]
