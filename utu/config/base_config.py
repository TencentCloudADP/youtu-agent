from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

ReprArgs: type = Iterable[tuple[str | None, Any]]

SECURE_FIELDS = ("api_key", "base_url")


def if_need_secure(key: str) -> bool:
    return any(f in key.lower() for f in SECURE_FIELDS)


def secure_repr(obj: ReprArgs) -> ReprArgs:
    for k, v in obj:
        if if_need_secure(k):
            yield k, "***"
        else:
            yield k, v


class ConfigBaseModel(BaseModel):
    """Base model for config, with secure repr"""

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in secure_repr(self.__repr_args__()))})"

    def model_dump(
        self,
        *,
        exclude_none: bool = True,  # avoid passing temperature=None to avoid SGLang error
        **kwargs,
    ) -> dict[str, Any]:
        return super().model_dump(exclude_none=exclude_none, **kwargs)
