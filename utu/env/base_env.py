import abc
import datetime

from agents import Tool


class BaseEnv:
    """Environment interface for agents."""

    @abc.abstractmethod
    def get_state(self) -> str:
        """Get the current state of the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_tools(self) -> list[Tool]:
        """Get the tools available in the environment."""
        raise NotImplementedError

    async def build(self):
        """Build the environment."""
        pass

    async def cleanup(self):
        """Cleanup the environment."""
        pass

    async def __aenter__(self):
        await self.build()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


class BasicEnv(BaseEnv):
    @staticmethod
    def get_time() -> str:
        return datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")

    def get_state(self) -> str:
        return ""

    async def get_tools(self) -> list[Tool]:
        return []
