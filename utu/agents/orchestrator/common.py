import json
from dataclasses import dataclass, field
from typing import Literal

from agents import TResponseInputItem

# from openai.types.responses import EasyInputMessageParam
from ..common import DataClassWithStreamEvents


@dataclass
class Task:
    agent_name: str
    task: str
    result: str | None = None
    is_last_task: bool = False  # whether this task is the last task of the plan


@dataclass
class Plan:
    input: str = ""
    analysis: str = ""
    tasks: list[Task] = field(default_factory=list)

    def format_tasks(self) -> str:
        tasks = [{"name": t.agent_name, "task": t.task} for t in self.tasks]
        return json.dumps(tasks, ensure_ascii=False)

    def format_plan(self) -> str:
        return f"<analysis>{self.analysis}</analysis>\n<plan>{self.format_tasks()}</plan>"


@dataclass
class OrchestratorStreamEvent:
    name: Literal[
        "plan.start",
        "plan.done",
        "task.start",
        "task.done",
    ]
    item: Plan | Task | None = None
    type: Literal["orchestrator_stream_event"] = "orchestrator_stream_event"


@dataclass
class Recorder(DataClassWithStreamEvents):
    # current main task
    input: str = ""  # current user input
    final_output: str = None  # final output to `input`. Naming consistent with @agents.RunResult
    trajectories: list = field(default_factory=list)  # trajs corresponding to `input`
    trace_id: str = ""

    # planning
    tasks: list[Task] = None
    current_task_id: int = 0

    # history
    history_plan: list[TResponseInputItem] = field(default_factory=list)
    history_messages: list[TResponseInputItem] = field(default_factory=list)

    def get_plan_str(self) -> str:
        return "\n".join([f"{i}. {t.task}" for i, t in enumerate(self.tasks, 1)])

    def get_trajectory_str(self) -> str:
        traj = []
        for t in self.tasks:
            if t.result is None:
                break
            traj.append(f"<task>{t.task}</task>\n<output>{t.result}</output>")
        return "\n".join(traj)

    def add_plan(self, plan: Plan) -> None:
        self.tasks = plan.tasks
        self.history_plan.extend(
            [
                {"role": "user", "content": f"<question>{plan.input}</question>"},
                {"role": "assistant", "content": plan.format_plan()},
            ]
        )

    def add_final_output(self, final_output: str) -> None:
        self.final_output = final_output
        self.history_messages.extend(
            [
                {"role": "user", "content": "self.input"},
                {"role": "assistant", "content": final_output},
            ]
        )

    def new(self, input: str = None, trace_id: str = None) -> "Recorder":
        """Create a new recorder with the same history -- for multi-turn chat."""
        new_recorder = Recorder(input=input, trace_id=trace_id)
        new_recorder.history_plan = self.history_plan.copy()
        new_recorder.history_messages = self.history_messages.copy()
        return new_recorder
