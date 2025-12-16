import pathlib

from agents import RunResultStreaming

from ...agents.llm_agent import LLMAgent
from ...config import AgentConfig
from ...utils import FileUtils
from .common import AnalysisResult, OrchestraStreamEvent, OrchestraTaskRecorder


class ReporterAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMAgent(model_config=config.reporter_model, name="reporter")
        self.template = self._get_template()

    @property
    def name(self) -> str:
        return self.config.reporter_config.get("name", "reporter")

    def _get_template(self):
        template_path = self.config.reporter_config.get("template_path", None)
        if template_path and pathlib.Path(template_path).exists():
            template_path = pathlib.Path(template_path)
        else:
            template_path = "agents/orchestra/reporter_sp.j2"
        return FileUtils.get_jinja_template(template_path)

    async def report(self, task_recorder: OrchestraTaskRecorder) -> AnalysisResult:
        """analyze the result of a subtask, return a report"""
        query = self.template.render(question=task_recorder.task, trajectory=task_recorder.get_trajectory_str())
        task_recorder._event_queue.put_nowait(OrchestraStreamEvent(name="report_start"))
        res = self.llm.run_streamed(query)
        await self._process_streamed(res, task_recorder)
        analysis_result = AnalysisResult(output=res.final_output)
        task_recorder._event_queue.put_nowait(OrchestraStreamEvent(name="report", item=analysis_result))
        return analysis_result

    async def _process_streamed(self, run_result_streaming: RunResultStreaming, task_recorder: OrchestraTaskRecorder):
        async for event in run_result_streaming.stream_events():
            task_recorder._event_queue.put_nowait(event)
