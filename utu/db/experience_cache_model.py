from typing import Any

from sqlalchemy import JSON, Integer
from sqlmodel import Column, Field, Float, SQLModel, String


class ExperienceCacheModel(SQLModel, table=True):
    __tablename__ = "cache_experience"

    id: int | None = Field(default=None, primary_key=True)

    experiment_name: str = Field(sa_column=Column(String))
    step: int = Field(sa_column=Column(Integer))
    epoch: int | None = Field(default=None, sa_column=Column(Integer))
    batch: int | None = Field(default=None, sa_column=Column(Integer))

    experiences: Any | None = Field(default=None, sa_column=Column(JSON))

    timestamp: float = Field(sa_column=Column(Float))
    datetime: str = Field(sa_column=Column(String))
