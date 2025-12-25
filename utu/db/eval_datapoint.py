import datetime
from typing import Any

from sqlmodel import JSON, Column, Field, SQLModel

from .utu_basemodel import UTUBaseModel


class DatasetSample(SQLModel, table=True):
    __tablename__ = "data"

    id: int | None = Field(default=None, primary_key=True)
    dataset: str = ""  # dataset name, for exp
    index: int | None = Field(default=None)  # The index of the datapoint in the dataset
    source: str = ""  # dataset name for mixed dataset
    source_index: int | None = Field(default=None)  # The index of the datapoint in the source dataset, if available

    question: str = ""
    answer: str | None = ""
    topic: str | None = ""
    level: int | None = 0  # hardness level of the question, if applicable
    file_name: str | None = ""  # for GAIA

    meta: Any | None = Field(
        default=None, sa_column=Column(JSON)
    )  # e.g. annotator_metadata in GAIA, extra_info in WebWalkerQA


class EvaluationSample(UTUBaseModel, SQLModel, table=True):
    __tablename__ = "evaluation_data"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime.datetime | None = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime | None = Field(default_factory=datetime.datetime.now)

    # 1) base info
    dataset: str = ""  # dataset name
    dataset_index: int | None = Field(default=None)
    source: str = ""
    raw_question: str = ""
    level: int | None = 0  # hardness level of the question, if applicable
    augmented_question: str | None = ""
    correct_answer: str | None = ""
    file_name: str | None = ""  # for GAIA
    meta: Any | None = Field(default=None, sa_column=Column(JSON))
    # 2) rollout
    trace_id: str | None = Field(default=None)
    trace_url: str | None = Field(default=None)
    response: str | None = Field(default=None)
    time_cost: float | None = Field(default=None)  # time cost in seconds
    trajectory: str | None = Field(default=None)  # deprecated, use trajectories instead for multi-agents
    trajectories: str | None = Field(default=None)
    # 3) judgement
    extracted_final_answer: str | None = Field(default=None)
    judged_response: str | None = Field(default=None)
    reasoning: str | None = Field(default=None)
    correct: bool | None = Field(default=None)
    reward: float | None = Field(default=None)  # 0-1
    confidence: int | None = Field(default=None)
    # id
    exp_id: str = Field(default="default")
    stage: str = "init"  # Literal["init", "rollout", "judged]

    def model_dump(self, *args, **kwargs):
        keys = [
            "exp_id",
            "dataset",
            "dataset_index",
            "source",
            "level",
            "raw_question",
            "correct_answer",
            "file_name",
            "stage",
            "trace_id",
            "response",
            "time_cost",
            "trajectory",
            "trajectories",
            "judged_response",
            "correct",
            "reward",
            "confidence",
        ]
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}
