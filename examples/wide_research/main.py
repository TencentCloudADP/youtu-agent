# pylint: disable=line-too-long
# ruff: noqa: E501
"""
https://manus.im/blog/introducing-wide-research
"""

import asyncio
import json
import pathlib
import traceback

from agents import function_tool

from utu.agents import SimpleAgent
from utu.config import ConfigLoader
from utu.tools import SearchToolkit
from utu.utils import AgentsUtils, FileUtils, schema_to_basemodel

PROMPTS = FileUtils.load_yaml(pathlib.Path(__file__).parent / "prompts.yaml")
SEARCH_TOOLKIT = SearchToolkit(ConfigLoader.load_toolkit_config("search"))
CONCURRENCY = 20


@function_tool(strict_mode=False)
async def search_wide(task: str, subtasks: list[str], output_schema: dict, output_fn: str) -> str:
    """Perform structured wide research. Given task with several search subtasks, this tool will perform research simultaneously.
    It is helpful when you need to collect several homogeneous information of the same topic.

    NOTEs:
    - Use this tool when the root task has >= 5 subtasks and the subtasks are homogeneous (have the same output schema).
    - The output will be saved to a jsonl file.

    Args:
        task (str): The root task to perform research on.
        subtasks (list[str]): Homogeneous subtasks contained in the root task.
        output_schema (dict): The desired output format of each subtask, MUST be valid JSON Schema. e.g.
          {"properties": {"provider": {"description": "The model provider", "title": "Provider", "type": "string"}, "model_name": {"description": "The model name", "title": "Model Name", "type": "string"}, "context_window": {"description": "The context window", "type": "integer"}, "required": ["provider", "model_name", "context_window"], "title": "LLM", "type": "object"}
        output_fn (str): The file name to save the output, in JSONL format. e.g. `output.jsonl`
    """
    # NOTE: we don't use output_type here for compatibility of APIs without structured output feature, you can open the output_type if your LLM provider supports it
    output_type = schema_to_basemodel(output_schema)  # noqa: F841
    print(f"Processing {len(subtasks)} subtasks for task: {task}\nOutput schema: {output_schema}")
    try:
        searcher = SimpleAgent(
            name="SearcherAgent",
            instructions=PROMPTS["searcher"].format(schema=output_schema),
            tools=SEARCH_TOOLKIT.get_tools_in_agents(),
            # output_type=AgentOutputSchema(output_type=output_type, strict_json_schema=False),
        )
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def run_with_semaphore(idx: int, task: str) -> str:
            async with semaphore:
                try:
                    async with searcher:
                        res = await searcher.run(task)
                        final_output = res.get_run_result().final_output
                    print(f"{idx}: `{task}` task finished!")
                    return final_output
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()
                    return f"Error: {e}"

        results = await asyncio.gather(*[run_with_semaphore(i, subtask) for i, subtask in enumerate(subtasks)])
        output_fn = pathlib.Path(__file__).parent / output_fn
        with open(output_fn, "w") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        with open(output_fn) as f:
            return f"Results saved to {output_fn}:\n" + f.read()
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"


class WideResearch:
    async def build(self):
        self.planner_agent = SimpleAgent(
            name="PlannerAgent",
            instructions=PROMPTS["planner"],
            tools=[search_wide] + SEARCH_TOOLKIT.get_tools_in_agents(),
        )

    async def run(self, task: str):
        async with self.planner_agent as planner:
            result = await planner.run(task)
            output = result.get_run_result().final_output
            return output

    async def run_streamed(self, task: str):
        async with self.planner_agent as planner:
            result = planner.run_streamed(task)
            await AgentsUtils.print_stream_events(result.stream_events())
            output = result.final_output
            return output


TASK = "Find the outstanding papers of ACL 2025, extract their title, author list, keywords, abstract, url. Return a markdown table"


async def main():
    wide_research = WideResearch()
    await wide_research.build()
    query = input("What would you like to research? ")
    query = query.strip() or TASK
    print(f"Processing task: {query}")
    result = await wide_research.run_streamed(query)
    print(f"{'-' * 80}\n{result}\n{'-' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
