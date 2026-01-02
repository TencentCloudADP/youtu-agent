# ruff: noqa
"""
Example scritp that defines a customized planner agent loop.

Ref:
    In-the-Flow Agentic System Optimization for Effective Planning and Tool Use, https://arxiv.org/abs/2510.05592
"""
import asyncio
import collections
import itertools
from copy import deepcopy
import json
from agents import FunctionTool, trace
from agents.models.chatcmpl_converter import Converter

from utu.agents import LLMAgent, SimpleAgent
from utu.config import ConfigLoader
from utu.tools import TOOLKIT_MAP
from utu.utils import LLMOutputParser, AgentsUtils

import re
from typing import Dict, Optional




def parse_response_sections(response_text: str, tags: list=[]) -> Dict[str, Optional[str]]:
    """
    从结构化响应文本中解析出 justification, context, subgoal, tools 四个部分
    
    Args:
        response_text: 包含特定标签的原始文本
        
    Returns:
        包含四个字段的字典，如果某个字段未找到则为 None
    """
    # 定义要解析的标签
    result = {}
    
    for tag in tags:
        # 使用非贪婪匹配和 DOTALL 模式（支持跨行内容）
        # 同时忽略标签前后可能存在的空白字符
        pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            result[tag] = match.group(1).strip()
        else:
            result[tag] = None
    
    return result



def parse_planner_response(response_text: str):
    tags = ["justification", "context", "subgoal", "tools"]
    sections = parse_response_sections(response_text, tags)
    tools = sections["tools"]
    if tools:
        tools = tools.split(",")
        tools = [t.strip() for t in tools]
        sections["tools"] = tools
    return sections


def parse_verifier_response(response_text: str):
    tags = ["justification", "conclusion"]
    sections = parse_response_sections(response_text, tags)
    return sections


def parse_answerer_response(response_text: str):
    tags = ["summary", "answer"]
    sections = parse_response_sections(response_text, tags)
    return sections


def parse_summarizer_response(response_text: str):
    tags = ["summary"]
    sections = parse_response_sections(response_text, tags)
    return sections


P_PLANNER = r"""You are a planner. You determine the optimal next step to address the query using available tools and previous context, then another agent will complete the next step for you.


Instructions:
- Analyze the current objective, the history of executed steps, and the capabilities of the available tools.
- Select the single most appropriate tool for the next action.
- Consider the specificity of the task (e.g., calculation vs. information retrieval).
- Consider the source of required information (e.g., general knowledge, mathematical computation, a
specific URL).
- Consider the limitations of each tool as defined in the metadata.
- Formulate a clear, concise, and achievable sub-goal that precisely defines what the selected tool should
accomplish.
- Provide all necessary context (e.g., relevant data, variable names, file paths, or URLs) so the tool can
execute its task without ambiguity.


Response Format:
<justification> Explain why the chosen tool(s) is(are) optimal for the sub-goal, referencing its capabilities and the task requirements. </justification>
<context> Provide all prerequisite information for the tool(s). </context>
<subgoal> Summarize the next objective with the tool(s). </subgoal>
<tools> State the exact name(s) of the selected tool(s). At least one tool should be selected. If multiple tools are selected, separate them with comma. <tools>


Rules:
- Select only one tool per step.
- The Sub-Goal must be directly and solely achievable with the selected tool.
- The Context section must contain all information the tool needs; do not assume implicit knowledge.
- The final response must end with the <justification> ... </justification>\n<context> ... </context>\n<subgoal> ... </subgoal>\n<tools> ... <tools> in that order. No additional text should follow.
""".strip()



P_VERIFIER = r"""You are a verifier that checks if the given query can be completed given the previous context (e.g., an accurate answer can be derived).
If not, more rounds of subgoal planning and tool execution will be performed.

Instructions:
- Review the query, the initial analysis, and the context of history actions and results.
- Does the accumulated information fully address all aspects of the query?
- Are there any unanswered sub-questions or missing pieces of information?
- Are there any inconsistencies or contradictions between different steps?
- Is any information ambiguous, potentially hallucinated, or in need of verification?
- Determine if any unused tools could provide critical missing information based on their metainfo.

Response Format:
<justification> If the context is sufficient to form a complete and accurate answer, explain why and conclude with "Conclusion: STOP". If more information is needed, clearly state what is missing, suggest which tool(s) could help, and conclude with "Conclusion: CONTINUE". </justification>
<conclusion> STOP/CONTINUE </conclusion>

Rules:
- The justification must be concise and directly tied to the query and context.
- The conclusion must be either exactly "STOP" or "CONTINUE".
- The final response must end with the <justification> ... </justification>\n<conclusion> ... </conclusion> in that order. No additional text should follow.
""".strip()


T_PLANNER = """Query: {query}
Available Tools: {available_tools}
Previous Steps (context): {previous_steps}"""


T_VERIFIER = """Query: {query}
Previous Steps (context): {previous_steps}"""


T_SUBAGENT = """You are a tool executor that completes the task assigned by the planner. Try you best to generate a precise command to call the candidate tool(s) to solve the task.

Task: {task}
Context: {context}

Instructions:
1. Analyze the tool required parameters from its metadata.
2. Choose the promising tool(s) that could address the task using the provided context.
2. Construct valid tool call format to ensure the tool name and parameters can be parsed correctly.
3. You must at least perform one tool call.
""".strip()



P_ANSWERER = r"""You are a solution generator. Provide a concise final answer to the query based on all provided context.

Instructions:
1. Carefully review the original user query and the complete history context (sequence of action steps and their results).
2. Synthesize the key findings from the history into a coherent narrative.
3. Construct a clear, step-by-step summary that explains how each step contributed to solving the query.
4. Provide a direct, precise, and standalone final answer to the original query.

Response Format:
<summary> A clear, step-by-step breakdown of how the query was addressed. For each step, state its purpose (e.g., "To verify X") and summarize its key results or findings in one sentence. </summary>
<answer> The direct and concise final answer to the query. This should be a self-contained statement that fully resolves the user question. </answer>

Rules:
- The summary should be informative but concise, focusing on the logical flow of the solution.
- The answer must be placed at the very end and be clearly identifiable.
- The response must follow the exact two-part structure above. It must end with the <summary> ... </summary>\n<answer> ... </answer> in that order. No additional text should follow.
"""

T_ANSWERER = """Query: {query}
Previous Steps (context): {previous_steps}"""


P_MEMORY_EVOLVER = r"""Summarize the history of previous steps (sequence of action steps and their results).

Instructions:
1. Analyze the history of previous steps.
2. Determine if each step is contributing to solving the query and how it does so.
3. Summarize the history briefly in one sentence, highlighting the conclusion of each step.

Response Format:
<summary> ... </summary>

Rules:
- The summary should be informative but concise, focusing on the logical flow of the solution.
- The response must follow the exact two-part structure above. It must end with the <summary> ... </summary>. No additional text should follow.
""".strip()


T_MEMORY_EVOLVER = """Query: {query}
Previous Steps (context): {previous_steps}"""



TOOLKIT_NAME_TO_TOOLS = {
    "direct_reply": ["direct_reply"],
    #### "search": ["search", "web_qa"],
    #### "python_executor": ["execute_python_code"],
    # "search": ["search_web"],
    # "extract_web_content": ["extract_web_content"],
    "wikilocal": ["search_wiki"],
    "codesnip": ["code_interpreter"],
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



async def summarize_memory(query, memory_evolver, system_prompt, memory, window_size=3):
    """
    Summarize your previous steps in the memory
    Args:
        query: the query
        memory_evolver: the memory evolver agent
        system_prompt: the system prompt
        memory: the memory (list of dicts)
        window_size: the window size
    Returns:
        messages_memory: the messages memory
        summary_dict: the summary dictionary
    """
    num_total_steps = len(memory)
    summary_dict = {}
    if len(memory) > window_size:
        # 如果记忆长度大于窗口大小，则总结窗口大小的记忆
        memory_summary = memory[:-window_size]
        _input = T_MEMORY_EVOLVER.format(query=query, previous_steps=json.dumps(memory_summary, ensure_ascii=False, indent=4))
        summaryagent_res = await memory_evolver.run(_input)
        summaryagent_output = parse_summarizer_response(response_text=summaryagent_res.final_output)
        assert all(k in summaryagent_output for k in ["summary"])
        # ---------------------------   包装memory输出内容 --------------------------- #
        messages_memory = Converter.items_to_messages(summaryagent_res.to_input_list())
        summary_dict[f"Step_1-Step_{num_total_steps - window_size}"] = summaryagent_output["summary"]
        messages_memory.insert(0, {"role": "system", "content": system_prompt})
        # 保留最新的window_size步记忆
        memory_complete = memory[-window_size:]
        for mem_idx, mem in enumerate(memory_complete):
            summary_dict[f"Step_{num_total_steps - window_size + mem_idx + 1}"] = mem

    else:
        # 如果记忆长度小于等于窗口大小，则不总结
        messages_memory = []
        # 保留所有记忆不做删除
        memory_complete = memory
        for mem_idx, mem in enumerate(memory_complete):
            summary_dict[f"Step_{mem_idx + 1}"] = mem
    return messages_memory, summary_dict




async def main_agent_loop(query: str, debug_mode: bool = False):
    """
    Args:
        query: the query
        debug_mode: whether to print debug information
    Returns:
        messages_by_turns: the messages by turns
        final_answer: the final answer
    """
    with trace("Customized planner agent loop"):
        model_config = ConfigLoader.load_model_config("base")

        planner = LLMAgent(model_config=model_config, name="custom_planner", instructions=P_PLANNER)
        verifier = LLMAgent(model_config=model_config, name="custom_verifier", instructions=P_VERIFIER)
        answerer = LLMAgent(model_config=model_config, name="custom_answergenerator", instructions=P_ANSWERER)
        memory_evolver = LLMAgent(model_config=model_config, name="custom_memory_evolver", instructions=P_MEMORY_EVOLVER)

        max_turns = 10
        available_tools_names = list(itertools.chain.from_iterable(TOOLKIT_NAME_TO_TOOLS.values()))
        available_tools_schemas = await get_tools(available_tools_names)
        available_tools = []
        for available_tool_schema_idx, available_tool_schema in enumerate(available_tools_schemas):
            tool_str = f"<tool{available_tool_schema_idx+1}>\nname: {available_tool_schema.name}\ndescription: {available_tool_schema.description}\n</tool{available_tool_schema_idx+1}>\n"
            available_tools.append(tool_str)
        available_tools = "\n\n" + "\n".join(available_tools) + "\n\n"
        print("> Available tools:", available_tools)
        # 裸记忆，不进行总结
        previous_steps = []  # memory
        stop_flag = False
        # message turn by turn
        messages_by_turns = []
        print(f"--- Starting agent loop ---\n{query}\n")

        for i in range(max_turns):
            # 记录当前轮的所有内容，包括memory、planner、tool_executor、verifier、answerer
            if debug_mode:
                for previous_step in previous_steps:
                    assert isinstance(previous_step, dict)
                for message_by_turn in messages_by_turns:
                    assert isinstance(message_by_turn, dict)
            messages_current_turn = {}
            if stop_flag:
                break
            print(f"--- Turn {i + 1} ---")
            # ---------------------------   首先处理memory --------------------------- #
            messages_memory, mem_history_dict = await summarize_memory(query, memory_evolver, P_MEMORY_EVOLVER, previous_steps, window_size=3)
            # 无tools 深拷贝messages_memory，避免后续修改影响原始memory
            messages_current_turn["memory"] = {"tools":[], "messages": deepcopy(messages_memory)}
            if debug_mode:
                print(f"> Memory history dict [initialization summary {i+1} turn]:", mem_history_dict)
                import pdb;pdb.set_trace();
            # ---------------------------   执行planner --------------------------- #
            _input = T_PLANNER.format(query=query, available_tools=available_tools, previous_steps=json.dumps(mem_history_dict, ensure_ascii=False, indent=4))
            planner_res = await planner.run(_input)
            planner_output = parse_planner_response(response_text=planner_res.final_output)
            # ---------------------------   包装planner输出内容 --------------------------- #
            messages_planner = Converter.items_to_messages(planner_res.to_input_list())
            messages_planner.insert(0, {"role": "system", "content": P_PLANNER})
            # 无tools
            messages_current_turn["planner"] = {"tools": [], "messages": messages_planner}
            print("> Planner output:", messages_planner)
            assert all(k in planner_output for k in ["justification", "context", "subgoal", "tools"])
            # ---------------------------   planner输出->历史队列 --------------------------- #
            current_step = {}
            # place-holder 当前的记忆内容
            current_step["subgoal"] = planner_output["subgoal"]
            current_step["tools"] = planner_output["tools"]
            mem_history_dict[f"Step_{len(previous_steps)+1}"] = current_step
            if debug_mode:
                print(f"> Memory history dict [planner output {i+1} turn]:", mem_history_dict)
                import pdb;pdb.set_trace();         
            # ---------------------------   执行tool executor --------------------------- #
            model = AgentsUtils.get_agents_model(**model_config.model_provider.model_dump())
            tools = await get_tools(planner_output["tools"])
            # ---------------------------   包装tools json格式 --------------------------- #
            tools_schema = []
            for tool in tools:
                if isinstance(tool, FunctionTool):
                    tool_schema_params = tool.params_json_schema
                    tool_schema = {
                        "type": "function",
                        "function":{
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool_schema_params
                        }
                    }
                    tools_schema.append(tool_schema)
            print("> Planner selected tools:", tools_schema)
            subagent = SimpleAgent(model=model, name="tool_executor", instructions=None, tools=tools, tool_use_behavior="stop_on_first_tool")
            _input = T_SUBAGENT.format(task=planner_output["subgoal"], context=planner_output["context"])
            subagent_res = await subagent.run(_input)
            subagent_output = subagent_res.final_output
            # ---------------------------   包装tools输出内容 --------------------------- #
            sub_agent_messages = Converter.items_to_messages(subagent_res.to_input_list())
            # 有tools
            messages_current_turn["tool_executor"] = {"tools": tools_schema, "messages": sub_agent_messages}
            sub_agent_messages_tools = [m for m in sub_agent_messages if ("role" in m) and (m["role"] in ["assistant", "tool"])]
            # ---------------------------   tool executor输出->历史队列 --------------------------- #
            current_step["tool_calls_and_responses"] = sub_agent_messages_tools
            mem_history_dict[f"Step_{len(previous_steps)+1}"] = deepcopy(current_step)
            if debug_mode:
                print(f"> Memory history dict [tool executor output {i+1} turn]:", mem_history_dict)
                import pdb;pdb.set_trace();
            print("> Tool executor output:", sub_agent_messages_tools)
            # ---------------------------   执行verifier --------------------------- #
            _input = T_VERIFIER.format(query=query, previous_steps=json.dumps(mem_history_dict, ensure_ascii=False, indent=4))
            verifier_res = await verifier.run(_input)
            verifier_output = parse_verifier_response(response_text=verifier_res.final_output)
            assert all(k in verifier_output for k in ["justification", "conclusion"])
            # ---------------------------   包装verifier输出内容 --------------------------- #
            verifier_messages = Converter.items_to_messages(verifier_res.to_input_list())
            verifier_messages.insert(0, {"role": "system", "content": P_VERIFIER})
            # 无tools
            messages_current_turn["verifier"] = {"tools": [], "messages": verifier_messages}
            # ---------------------------   verifier输出->历史队列 --------------------------- #
            current_step["verification_status"] = verifier_output["justification"]
            current_step["final_determination"] = verifier_output["conclusion"]
            mem_history_dict[f"Step_{len(previous_steps)+1}"] = deepcopy(current_step)
            if debug_mode:
                print(f"> Memory history dict [verifier output {i+1} turn]:", mem_history_dict)
                import pdb;pdb.set_trace();
            print("> Verifier output:", verifier_output)
            if verifier_output["conclusion"].upper() == "STOP":
                # 停止下一轮
                stop_flag = True
            previous_steps.append(deepcopy(current_step))
            messages_by_turns.append(deepcopy(messages_current_turn))

        if stop_flag:
            print("--- Early stop triggered by Stop flag ---")
        else:
            print("--- Max turns reached ---")
        # ---------------------------   执行answerer --------------------------- #
        # 存储所有历史记录 直接返回
        messages_memory, mem_history_dict = await summarize_memory(query, memory_evolver, P_MEMORY_EVOLVER, previous_steps, window_size=len(previous_steps))
        if debug_mode:
            print(f"> Memory history dict [answerer input {i+1} turn]:", mem_history_dict)
            import pdb;pdb.set_trace();
        _input = T_ANSWERER.format(query=query, previous_steps=json.dumps(mem_history_dict, ensure_ascii=False, indent=4))
        answerer_res = await answerer.run(_input)
        answerer_res_output = parse_answerer_response(response_text=answerer_res.final_output)
        assert all(k in answerer_res_output for k in ["summary", "answer"])
        # ---------------------------   包装answerer输出内容 --------------------------- #
        answerer_messages = Converter.items_to_messages(answerer_res.to_input_list())
        answerer_messages.insert(0, {"role": "system", "content": P_ANSWERER})
        # 无tools 直接存储最后一轮输出
        messages_by_turns.append({"answerer": {"tools": [], "messages": deepcopy(answerer_messages)}})
        print("> Final Answer:", answerer_res_output)
        return messages_by_turns, answerer_res_output["answer"]




if __name__ == "__main__":
    print()
    """
    response_text = '''
<justification>
The user is asking for current weather information in Tokyo, which is a real-time data request that falls outside my training knowledge. The web_search tool is specifically designed to retrieve up-to-date information like weather, news, or live data from the internet. Since the web features are disabled by the user, I must first inform them to enable the toggle before proceeding with the search.
</justification>

<context>
The user has manually disabled web features (Search & Open Url) in the input box settings. To fulfill the request for current Tokyo weather, the user needs to re-enable these features so that web_search can be used to retrieve live weather data.
</context>

<subgoal>
Inform the user that web features are currently disabled and request them to enable the toggle so that web_search can be used to fetch the current weather in Tokyo.
</subgoal>

<tools>
None (inform user to enable web features)
<tools>'''.strip()
    print(parse_planner_response(response_text=response_text))

    response_text = '''
<justification>
The query asks for current weather in Tokyo. No previous context or data about Tokyo's weather has been provided. To answer accurately, up-to-date external information is required. The web_search tool is designed for retrieving current data like weather, and it is currently disabled per user settings. Without any weather data available, the query cannot be satisfied.
Conclusion: CONTINUE
</justification>
<conclusion>CONTINUE</conclusion>'''.strip()
    print(parse_verifier_response(response_text=response_text))

    response_text = '''<summary>
No previous steps were taken; the query is being asked for the first time.
</summary>
<answer>
To provide current weather information for Tokyo, web search is required, but it is currently disabled in your settings. Please enable web features and ask again.
</answer>'''.strip()
    print(parse_answerer_response(response_text=response_text))
    # mimic the summarization process
    memory1 = []
    memory2 = []
    memory3 = []

    for step_idx in range(10):
        current_step = {}
        current_step["subgoal"] = f"subgoal_{step_idx+1}"
        current_step["tools"] = f"tools_{step_idx+1}"
        current_step["tool_calls_and_responses"] = f"tool_calls_and_responses_{step_idx+1}"
        current_step["verification_status"] = f"verification_status_{step_idx+1}"
        current_step["final_determination"] = f"final_determination_{step_idx+1}"
        if step_idx < 2:
            memory1.append(current_step)
        if step_idx <= 2:
            memory2.append(current_step)
        memory3.append(current_step)
    
    query = "This is a test example about subgoal decomposition."
    model_config = ConfigLoader.load_model_config("base")
    memory_evolver = LLMAgent(model_config=model_config, name="custom_memory_evolver", instructions=P_MEMORY_EVOLVER)
    mem_history = asyncio.run(summarize_memory(query, memory_evolver, memory3, window_size=len(memory3)))
    print(">>> 总结后的历史记录:\n", mem_history)
    """

    query = "Write a Python function to compute the Fibonacci sequence up to n, and then use it to find the 10th Fibonacci number."
    # query = "What is the middle name of Donald Trump?"
    # query = "What is the weather in Shanghai now?"
    print("Question:\n", query)
    messages_by_turns, final_answer = asyncio.run(main_agent_loop(query, debug_mode=False))
    print("Answer:\n", final_answer)
    save_jsonl_path = "/cfs_turbo/yuleiqin/Research/youtu-agent/examples/in_the_flow/debug_messages.json"
    with open(save_jsonl_path, "w") as fw:
        json.dump({"messages_by_turns": messages_by_turns, "final_answer": final_answer}, fw, ensure_ascii=False)

