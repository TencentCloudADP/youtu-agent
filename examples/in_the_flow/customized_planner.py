# ruff: noqa
"""
Example scritp that defines a customized planner agent loop.

Ref:
    In-the-Flow Agentic System Optimization for Effective Planning and Tool Use, https://arxiv.org/abs/2510.05592
"""
import asyncio
import collections
import itertools

from agents import FunctionTool, trace

from utu.agents import LLMAgent, SimpleAgent
from utu.config import ConfigLoader
from utu.tools import TOOLKIT_MAP
from utu.utils import LLMOutputParser, AgentsUtils



P_PLANNER = r"""You are a planner. You can give actions to subagents to complete user's query. 
You should decide next action based on the query and previous steps, and select tools for the subagent. 
You should response in the following JSON format, without any other text.

Output format: 
```json
{"action": "<summary of next step>", "description": "<detailed infos that pass to the subagent, helping him understand the full context of this task>", "tools": "<list of tool names for the subagent, at least one tool should be selected>"}
```
""".strip()




P_VERIFIER = r"""You are a verifier that checks the answer for a given query. Given a query, an answer, and previous steps, you need to decide next step: either `STOP` if the answer is correct, or `CONTINUE` if more information is needed.
You should response in the following JSON format, without any other text.

Output format: 
```json
{"think": "<your rationale to conduct the final conclusion>", "conclusion": "STOP/CONTINUE"}
```
""".strip()




T_PLANNER = """Query: {query}
Available Tools: {available_tools}
Previous Steps: {previous_steps}"""

T_VERIFIER = """Query: {query}
Answer: {answer}
Previous Steps: {previous_steps}"""

T_SUBAGENT = """
You act as a subagent to help complete the task assigned by the planner. Try you best to use the provided tools to give the final answer.

Task: {task}
Context: {context}
""".strip()




TOOLKIT_NAME_TO_TOOLS = {
    "search": ["search", "web_qa"],
    "python_executor": ["execute_python_code"],
}
async def get_tools(selected_tool_name: list[str]) -> list[FunctionTool]:
    """Get FunctionTools based on selected tool names."""
    tool_name_to_toolkit = {}
    for tk_name, tools in TOOLKIT_NAME_TO_TOOLS.items():
        for tool in tools:
            tool_name_to_toolkit[tool] = tk_name
    selected_toolkits = collections.defaultdict(list)
    for name in selected_tool_name:
        tk_name = tool_name_to_toolkit[name]
        selected_toolkits[tk_name].append(name)
    tools = []
    for tk_name, tool_names in selected_toolkits.items():
        tk_config = ConfigLoader.load_toolkit_config(tk_name)
        tk_config.activated_tools = tool_names
        toolkit_instance = TOOLKIT_MAP[tk_name](config=tk_config)
        tools.extend(toolkit_instance.get_tools_in_agents())
    return tools


async def main_agent_loop(query: str):
    with trace("Customized planner agent loop"):
        model_config = ConfigLoader.load_model_config("base")

        planner = LLMAgent(model_config=model_config, name="custom_planner", instructions=P_PLANNER)
        verifier = LLMAgent(model_config=model_config, name="custom_verifier", instructions=P_VERIFIER)

        max_turns = 5
        available_tools = list(itertools.chain.from_iterable(TOOLKIT_NAME_TO_TOOLS.values()))
        previous_steps = []  # memory
        print(f"--- Starting agent loop ---\n{query}\n")
        for i in range(max_turns):
            print(f"--- Turn {i + 1} ---")
            _input = T_PLANNER.format(query=query, available_tools=available_tools, previous_steps=previous_steps)
            planner_res = await planner.run(_input)
            planner_output = LLMOutputParser.extract_code_json(planner_res.final_output)
            assert all(k in planner_output for k in ["action", "description", "tools"])
            previous_steps.append({"role": "planner", "content": planner_output})
            print("> Planner output:", planner_output)

            model = AgentsUtils.get_agents_model(**model_config.model_provider.model_dump()),
            tools = await get_tools(planner_output["tools"])
            subagent = SimpleAgent(model=model, name="subagent", instructions=None, tools=tools, tool_use_behavior="stop_on_first_tool")
            _input = T_SUBAGENT.format(task=planner_output["action"], context=planner_output["description"])
            subagent_res = await subagent.run(_input)
            # stop on first tool call返回的final_output就是tool response
            subagent_output = subagent_res.final_output
            messages = subagent_res.to_input_list()
            messages_tools = [m for m in messages if m["type"] in ["function_call", "function_call_output"]]
            previous_steps.append({"role": "subagent", "content": subagent_output})
            print("> Subagent output:", subagent_output)
            import pdb;pdb.set_trace();
            _input = T_VERIFIER.format(query=query, answer=subagent_output, previous_steps=previous_steps)
            verifier_res = await verifier.run(_input)
            verifier_output = LLMOutputParser.extract_code_json(verifier_res.final_output)
            assert all(k in verifier_output for k in ["think", "conclusion"])
            previous_steps.append({"role": "verifier", "content": verifier_output})
            print("> Verifier output:", verifier_output)
            if verifier_output["conclusion"].upper() == "STOP":
                print(">> Final Answer:", subagent_output)
                return subagent_output
        print("Max turns reached. Final Answer:", subagent_output)
        return subagent_output


if __name__ == "__main__":
    query = "Write a Python function to compute the Fibonacci sequence up to n, and then use it to find the 10th Fibonacci number."
    asyncio.run(main_agent_loop(query))
