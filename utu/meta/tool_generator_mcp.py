"""
Experimental feature of tool generation

generation workflow:
1. query -> requirement (function schema);
2. schema -> implementation(python code);
3. schema+code -> manifest
4. postprocess

- [x] add clarification tool
- [x] debugger agent
- [ ] doc for coding-agent integration (e.g. Claude Code)
- [ ] toolkit hub?
"""

import asyncio
import pathlib
import subprocess
from dataclasses import dataclass, field

from agents import RunResultStreaming, trace

from ..agents import SimpleAgent
from ..agents.common import DataClassWithStreamEvents, QueueCompleteSentinel
from ..utils import DIR_ROOT, FileUtils, LLMOutputParser, PrintUtils, get_logger

logger = get_logger(__name__)


@dataclass
class TaskRecorder(DataClassWithStreamEvents):
    name: str = field(default_factory=str)
    description: str = field(default_factory=str)
    implementation: str = field(default_factory=str)
    manifest: dict = field(default_factory=dict)
    # class_name, requirements, methods, etc
    output_file: str = field(default_factory=str)


class ToolGenerator:
    def __init__(self, auto_debug: bool = True):
        self.auto_debug = auto_debug
        self.prompts = FileUtils.load_prompts("meta/tool_generator_mcp.yaml")
        self.llm = SimpleAgent(
            name="tool_generator",
            instructions="You are a Python software engineer assistant.",
            toolkits=["user_interaction", "search"],
        )
        self.debugger = SimpleAgent(config="meta/tool_generator_mcp_debugger") if auto_debug else None
        self.output_dir = DIR_ROOT / "configs/tools/generated"
        self.output_dir.mkdir(exist_ok=True)

    async def run(self, user_input: str):
        self.llm.clear_input_items()
        with trace("tool_generator"):
            task_recorder = TaskRecorder()
            # step 1: generate requirements
            await self.step1(task_recorder, user_input)
            # step 2: generate implementation
            await self.step2(task_recorder)
            # step 3: generate manifest
            await self.step3(task_recorder)
            # postprocess
            self.postprocess(task_recorder)
        print(f"Generated tool config saved to {task_recorder.output_file}")

    def run_streamed(self, user_input: str) -> TaskRecorder:
        task_recorder = TaskRecorder()
        task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder, user_input))
        return task_recorder

    async def _start_streaming(self, task_recorder: TaskRecorder, user_input: str):
        with trace("tool_generator"):
            try:
                await self.step1(task_recorder, user_input)
                await self.step2(task_recorder)
                await self.step3(task_recorder)
                self.postprocess(task_recorder)

                # Auto debug if enabled
                if self.auto_debug:
                    await self.run_debug(task_recorder=task_recorder, workspace_dir=task_recorder.name)
            except Exception as e:
                task_recorder._is_complete = True
                task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
                raise e

        task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
        task_recorder._is_complete = True

    async def step1(self, task_recorder: TaskRecorder, user_input: str) -> None:
        async with self.llm as agent:
            query = FileUtils.get_jinja_template_str(self.prompts["STEP_1_REQUIREMENT"]).render(user_request=user_input)
            res = agent.run_streamed(query, save=True)
            await self._process_streamed(res, task_recorder)
            parsed_res = LLMOutputParser.extract_code_json(res.final_output)
            task_recorder.name = parsed_res["name"]
            task_recorder.description = parsed_res["description"]

    async def step2(self, task_recorder: TaskRecorder) -> None:
        async with self.llm as agent:
            query = FileUtils.get_jinja_template_str(self.prompts["STEP_2_IMPLEMENTATION"]).render()
            res = agent.run_streamed(query, save=True)
            await self._process_streamed(res, task_recorder)
            task_recorder.implementation = LLMOutputParser.extract_code_python(res.final_output)

    async def step3(self, task_recorder: TaskRecorder) -> None:
        async with self.llm as agent:
            query = FileUtils.get_jinja_template_str(self.prompts["STEP_3_MANIFEST"]).render()
            res = agent.run_streamed(query, save=True)
            await self._process_streamed(res, task_recorder)
            task_recorder.manifest = LLMOutputParser.extract_code_json(res.final_output)

    def postprocess(self, task_recorder: TaskRecorder) -> None:
        name = task_recorder.name
        odir = self.output_dir / name
        odir.mkdir(exist_ok=True)
        with open(odir / "runner.py", "w") as f:
            f.write(task_recorder.implementation)
        with open(odir / "main.py", "w") as f:
            f.write(self.prompts["TEMPLATE_MAIN"])
        with open(odir / "manifest.json", "w") as f:
            f.write(
                FileUtils.get_jinja_template_str(self.prompts["TEMPLATE_MANIFEST"]).render(
                    name=name,
                    class_name=task_recorder.manifest["class_name"],
                    requirements=PrintUtils.format_json(task_recorder.manifest["requirements"]),
                    methods=PrintUtils.format_json(task_recorder.manifest["methods"]),
                    api_keys=PrintUtils.format_json(task_recorder.manifest["api_keys"]),
                )
            )
        with open(odir / "requirements.txt", "w") as f:
            f.write("\n".join(task_recorder.manifest["requirements"] + ["mcp"]))  # mcp is required
        config_fn = self.output_dir / f"{name}.yaml"
        with config_fn.open("w") as f:
            f.write(
                FileUtils.get_jinja_template_str(self.prompts["TEMPLATE_CONFIG"]).render(
                    name=name,
                )
            )
        task_recorder.output_file = str(config_fn)

        # init the environment
        self._init_environment(odir)

    def _init_environment(self, odir):
        """Initialize the virtual environment and install requirements."""
        subprocess.run(
            f"cd {odir} && uv venv && . .venv/bin/activate && uv pip install -r requirements.txt",
            shell=True,
            check=True,
        )

    async def _process_streamed(self, run_result_streaming: RunResultStreaming, task_recorder: TaskRecorder):
        async for event in run_result_streaming.stream_events():
            task_recorder._event_queue.put_nowait(event)

    async def run_debug(self, workspace_dir: str, task_recorder: TaskRecorder = None) -> TaskRecorder:
        task_recorder = task_recorder or TaskRecorder(name=workspace_dir)
        res = self.run_debug_streamed(workspace_dir)
        await self._process_streamed(res, task_recorder)
        return task_recorder

    def run_debug_streamed(self, workspace_dir: str) -> TaskRecorder:
        """Run debug on an existing tool workspace directory.

        Args:
            workspace_dir: Path to the tool workspace directory (relative to configs/tools/generated/)
        """
        if not self.debugger:
            self.debugger = SimpleAgent(config="meta/tool_generator_mcp_debugger")

        if isinstance(workspace_dir, str):
            workspace_dir = pathlib.Path(DIR_ROOT / "configs/tools/generated" / workspace_dir)
        self.debugger.setup_workspace(workspace_dir)
        return self.debugger.run_streamed(f"当前目录: {workspace_dir}")
