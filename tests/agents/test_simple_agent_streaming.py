import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from agents import AgentUpdatedStreamEvent, Model
from mcp import McpError

from utu.agents.simple_agent import SimpleAgent
from utu.config import AgentConfig, ToolkitConfig


@pytest.fixture
async def agent(monkeypatch):
    agent = SimpleAgent(config=AgentConfig(), model=Mock(spec=Model))
    monkeypatch.setattr("utu.agents.simple_agent.AgentsUtils.get_current_trace", lambda: object())
    db_add = Mock(side_effect=AssertionError("Unexpected database write"))
    monkeypatch.setattr("utu.agents.simple_agent.DBService.add", db_add)
    yield agent
    if agent.env is not None:
        await agent.cleanup()
    db_add.assert_not_called()


async def collect_events(recorder):
    return [event async for event in recorder.stream_events()]


async def cleanup_tasks(*tasks):
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_initialization_failure_wakes_waiting_consumer(agent, monkeypatch):
    build_started = asyncio.Event()
    release_build = asyncio.Event()
    error = RuntimeError("Tool initialization failed")

    async def failing_build(trace_id):
        build_started.set()
        await release_build.wait()
        raise error

    monkeypatch.setattr(agent, "build", failing_build)
    recorder = agent.run_streamed("hello", log_to_db=False)
    consumer = asyncio.create_task(collect_events(recorder))
    try:
        await asyncio.wait_for(build_started.wait(), timeout=5)
        await asyncio.sleep(0)
        assert not consumer.done()
        release_build.set()
        done, _ = await asyncio.wait([consumer], timeout=5)
        assert consumer in done, "Stream consumer remained blocked after initialization failed"
        with pytest.raises(RuntimeError) as raised:
            consumer.result()
        assert raised.value is error
        assert recorder._is_complete
    finally:
        await cleanup_tasks(consumer, recorder._run_impl_task)


async def test_mcp_startup_failure_reaches_stream_consumer(agent):
    agent.config.toolkits = {
        "broken_mcp": ToolkitConfig(
            name="broken_mcp",
            mode="mcp",
            mcp_transport="stdio",
            config={"command": sys.executable, "args": ["-c", "import sys; sys.exit(3)"]},
        )
    }
    recorder = agent.run_streamed("hello", log_to_db=False)
    consumer = asyncio.create_task(collect_events(recorder))
    try:
        done, _ = await asyncio.wait([consumer], timeout=5)
        assert consumer in done, "Stream consumer remained blocked after MCP startup failed"
        with pytest.raises(McpError, match="Connection closed") as raised:
            consumer.result()
        assert raised.value is recorder._run_impl_task.exception()
        assert recorder._is_complete
    finally:
        await cleanup_tasks(consumer, recorder._run_impl_task)


@pytest.mark.parametrize("initialized", [False, True])
async def test_successful_stream_preserves_events_and_history(agent, monkeypatch, initialized):
    if initialized:
        await agent.build()
    build = AsyncMock(wraps=agent.build)
    monkeypatch.setattr(agent, "build", build)
    history = [{"role": "assistant", "content": "done"}]
    events = []

    async def stream_events():
        event = AgentUpdatedStreamEvent(new_agent=agent.current_agent)
        events.append(event)
        await asyncio.sleep(0)
        yield event

    result = SimpleNamespace(
        stream_events=stream_events,
        final_output="done",
        to_input_list=lambda: history,
    )

    def run_streamed(**kwargs):
        result.last_agent = kwargs["starting_agent"]
        return result

    runner = Mock(run_streamed=Mock(side_effect=run_streamed))
    monkeypatch.setattr("utu.agents.simple_agent.get_runner", lambda name: runner)
    recorder = agent.run_streamed("hello", save=True, log_to_db=False)
    consumer = asyncio.create_task(collect_events(recorder))
    try:
        done, _ = await asyncio.wait([consumer], timeout=5)
        assert consumer in done
        assert consumer.result() == events
        assert len(events) == 1
        assert recorder.final_output == "done"
        assert recorder.raw_run_results == [result]
        assert agent.input_items == history
        assert agent.current_agent is result.last_agent
        assert build.await_count == (0 if initialized else 1)
        assert runner.run_streamed.call_args.kwargs["input"] == [{"content": "hello", "role": "user"}]
    finally:
        await cleanup_tasks(consumer, recorder._run_impl_task)


async def test_runner_failure_reaches_stream_consumer(agent, monkeypatch):
    error = RuntimeError("Runner failed")
    events = []

    async def stream_events():
        event = AgentUpdatedStreamEvent(new_agent=agent.current_agent)
        events.append(event)
        yield event
        await asyncio.sleep(0)
        raise error

    runner = Mock(run_streamed=Mock(return_value=SimpleNamespace(stream_events=stream_events)))
    monkeypatch.setattr("utu.agents.simple_agent.get_runner", lambda name: runner)
    recorder = agent.run_streamed("hello", log_to_db=False)
    received = []

    async def consume():
        async for event in recorder.stream_events():
            received.append(event)

    consumer = asyncio.create_task(consume())
    try:
        done, _ = await asyncio.wait([consumer], timeout=5)
        assert consumer in done
        with pytest.raises(RuntimeError) as raised:
            consumer.result()
        assert raised.value is error
        assert received == events
        assert len(received) == 1
        assert recorder._is_complete
    finally:
        await cleanup_tasks(consumer, recorder._run_impl_task)
