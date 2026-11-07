from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import pytest
from agents import Model, ModelResponse, Usage, trace
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from utu.agents.llm_agent import LLMAgent
from utu.config import ModelConfigs
from utu.config.model_config import ModelProviderConfig
from utu.db.trajectory_model import TrajectoryModel
from utu.utils import AgentsUtils


class LocalModel(Model):
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            output=[
                ResponseOutputMessage(
                    id="msg_test",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="local answer", annotations=[])],
                )
            ],
            usage=Usage(),
            response_id="resp_test",
        )

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError
        yield


@pytest.mark.parametrize("input", ["hello", [{"role": "user", "content": "hello"}]])
@pytest.mark.parametrize("trace_id", [None, "trace_" + "1" * 32])
async def test_run_preserves_recorder_metadata(monkeypatch: pytest.MonkeyPatch, input: Any, trace_id: str | None):
    monkeypatch.setattr(AgentsUtils, "get_agents_model", lambda **kwargs: LocalModel())
    monkeypatch.setattr("utu.agents.llm_agent.trace", partial(trace, disabled=True))
    agent = LLMAgent(ModelConfigs(model_provider=ModelProviderConfig(model="offline-test")))

    recorder = await agent.run(input) if trace_id is None else await agent.run(input, trace_id=trace_id)

    assert recorder.input is input
    assert recorder.task == (input if isinstance(input, str) else "")
    if trace_id is not None:
        assert recorder.trace_id == trace_id
    else:
        assert recorder.trace_id.startswith("trace_")
        assert len(recorder.trace_id) == 38
    assert recorder._is_complete is False
    assert recorder._stored_exception is None
    assert recorder._run_impl_task is None
    assert recorder._event_queue.empty()
    assert recorder.final_output == "local answer"
    assert recorder.get_run_result().input is input
    assert recorder.to_input_list()[0] == {"role": "user", "content": "hello"}
    assert recorder.trajectories[0]["trajectory"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "local answer"},
    ]

    trajectory = TrajectoryModel.from_task_recorder(recorder)
    assert trajectory.trace_id == recorder.trace_id
    assert trajectory.d_output == "local answer"
    if isinstance(input, str):
        assert trajectory.d_input == input
