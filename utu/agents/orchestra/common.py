from dataclasses import dataclass, field
from typing import Literal

from agents import RunResultStreaming

from ..common import DataClassWithStreamEvents, TaskRecorder


@dataclass
class AgentInfo:
    """Subagent information (for planner)"""

    name: str
    desc: str
    strengths: str
    weaknesses: str


@dataclass
class Subtask:
    agent_name: str
    task: str
    completed: bool | None = None


@dataclass
class CreatePlanResult(DataClassWithStreamEvents):
    analysis: str = ""
    todo: list[Subtask] = field(default_factory=list)

    @property
    def trajectory(self):
        todos_str = []
        for i, subtask in enumerate(self.todo, 1):
            todos_str.append(f"{i}. {subtask.task} ({subtask.agent_name})")
        todos_str = "\n".join(todos_str)
        return {
            "agent": "planner",
            "trajectory": [
                {"role": "assistant", "content": self.analysis},
                {"role": "assistant", "content": todos_str},
            ],
        }


@dataclass
class WorkerResult(DataClassWithStreamEvents):
    task: str = ""
    output: str = ""
    trajectory: dict = field(default_factory=dict)

    stream: RunResultStreaming | None = None


@dataclass
class AnalysisResult(DataClassWithStreamEvents):
    output: str = ""

    @property
    def trajectory(self):
        return {"agent": "analysis", "trajectory": [{"role": "assistant", "content": self.output}]}


@dataclass
class OrchestraTaskRecorder(TaskRecorder):
    plan: CreatePlanResult = field(default=None)
    task_records: list[WorkerResult] = field(default_factory=list)

    def set_plan(self, plan: CreatePlanResult):
        self.plan = plan
        self.trajectories.append(plan.trajectory)

    def add_worker_result(self, result: WorkerResult):
        self.task_records.append(result)
        self.trajectories.append(result.trajectory)

    def add_reporter_result(self, result: AnalysisResult):
        self.trajectories.append(result.trajectory)

    def get_plan_str(self) -> str:
        return "\n".join([f"{i}. {t.task}" for i, t in enumerate(self.plan.todo, 1)])

    def get_trajectory_str(self) -> str:
        return "\n".join(
            [
                f"<subtask>{t.task}</subtask>\n<output>{r.output}</output>"
                for i, (r, t) in enumerate(zip(self.task_records, self.plan.todo, strict=False), 1)
            ]
        )


@dataclass
class OrchestraStreamEvent:
    name: Literal["plan_start", "plan", "worker", "report_start", "report"]
    item: CreatePlanResult | WorkerResult | AnalysisResult | None = None
    type: Literal["orchestra_stream_event"] = "orchestra_stream_event"
