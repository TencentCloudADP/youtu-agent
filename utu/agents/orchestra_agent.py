"""
- [x] support streaming for planner & reporter
"""

import asyncio

from agents import trace

from ..config import AgentConfig, ConfigLoader
from ..utils import AgentsUtils, get_logger
from .common import QueueCompleteSentinel
from .orchestra import (
    AnalysisResult,
    CreatePlanResult,
    OrchestraTaskRecorder,
    PlannerAgent,
    ReporterAgent,
    SimpleWorkerAgent,
    Subtask,
    WorkerResult,
)

logger = get_logger(__name__)


class OrchestraAgent:
    def __init__(self, config: AgentConfig | str):
        """Initialize the orchestra agent"""
        if isinstance(config, str):
            config = ConfigLoader.load_agent_config(config)
        self.config = config
        # init subagents
        self.planner_agent = PlannerAgent(config)
        self.worker_agents = self._setup_workers()
        self.reporter_agent = ReporterAgent(config)

    def set_planner(self, planner: PlannerAgent):
        self.planner_agent = planner

    def _setup_workers(self) -> dict[str, SimpleWorkerAgent]:
        workers = {}
        for name, config in self.config.workers.items():
            assert config.type == "simple", f"Only support SimpleAgent as worker in orchestra agent, get {config}"
            workers[name] = SimpleWorkerAgent(config=config)
        return workers

    async def run(self, input: str, trace_id: str = None) -> OrchestraTaskRecorder:
        task_recorder = self.run_streamed(input, trace_id)
        async for _ in task_recorder.stream_events():
            pass
        return task_recorder

    def run_streamed(self, input: str, trace_id: str = None) -> OrchestraTaskRecorder:
        # TODO: error_tracing
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        task_recorder = OrchestraTaskRecorder(task=input, trace_id=trace_id)
        # Kick off the actual agent loop in the background and return the streamed result object.
        task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder))
        return task_recorder

    async def _start_streaming(self, task_recorder: OrchestraTaskRecorder):
        with trace(workflow_name="orchestra_agent", trace_id=task_recorder.trace_id):
            try:
                await self.plan(task_recorder)

                for task in task_recorder.plan.todo:
                    # print(f"> processing {task}")
                    worker_agent = self.worker_agents[task.agent_name]
                    await worker_agent.build()
                    result_streaming = worker_agent.work_streamed(task_recorder, task)
                    async for event in result_streaming.stream.stream_events():
                        task_recorder._event_queue.put_nowait(event)
                    result_streaming.output = result_streaming.stream.final_output
                    result_streaming.trajectory = AgentsUtils.get_trajectory_from_agent_result(result_streaming.stream)
                    task_recorder.add_worker_result(result_streaming)
                    # print(f"< processed {task}")

                await self.report(task_recorder)

                task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
                task_recorder._is_complete = True
            except Exception as e:
                task_recorder._is_complete = True
                task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
                raise e

    async def plan(self, task_recorder: OrchestraTaskRecorder) -> CreatePlanResult:
        """Step1: Plan"""
        plan = await self.planner_agent.create_plan(task_recorder)
        assert all(t.agent_name in self.worker_agents for t in plan.todo), (
            f"agent_name in plan.todo must be in worker_agents, get {plan.todo}"
        )
        logger.info(f"plan: {plan}")
        task_recorder.set_plan(plan)
        return plan

    async def work(self, task_recorder: OrchestraTaskRecorder, task: Subtask) -> WorkerResult:
        """Step2: Work"""
        worker_agent = self.worker_agents[task.agent_name]
        result = await worker_agent.work(task_recorder, task)
        task_recorder.add_worker_result(result)
        return result

    async def report(self, task_recorder: OrchestraTaskRecorder) -> AnalysisResult:
        """Step3: Report"""
        analysis_result = await self.reporter_agent.report(task_recorder)
        task_recorder.add_reporter_result(analysis_result)
        task_recorder.set_final_output(analysis_result.output)
        return analysis_result
