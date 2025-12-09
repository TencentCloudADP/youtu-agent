import abc
from typing import Any

from .common import TaskRecorder


class BaseAgent:
    name: str

    @abc.abstractmethod
    async def run(self, input: Any, trace_id: str = None, **kwargs) -> TaskRecorder:
        raise NotImplementedError

    async def build(self):
        pass

    async def cleanup(self):
        pass