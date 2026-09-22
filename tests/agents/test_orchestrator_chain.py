import pytest

from utu.agents.common import TaskRecorder
from utu.agents.orchestrator.chain import ChainPlanner
from utu.agents.orchestrator.common import Recorder


def test_parse_reports_missing_plan_block() -> None:
    planner = object.__new__(ChainPlanner)
    recorder = Recorder(input="summarize the report")

    with pytest.raises(AssertionError, match="No tasks parsed from plan"):
        planner._parse("<analysis>the response was truncated</analysis>", recorder)


def test_task_recorder_defaults_input_to_list() -> None:
    assert TaskRecorder().input == []
