"""Minimal one-shot runner for uTu-Agent.

This mirrors the core invocation flow used in `test.py`:
ConfigLoader.load_agent_config(...) -> get_agent(...) -> await agent.build() -> await agent.run(...)

"""

import argparse
import asyncio
from agents import FunctionTool, trace
from agents.models.chatcmpl_converter import Converter
from utu.agents import get_agent
from utu.config import ConfigLoader
import json



async def main() -> None:
    query = '''Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters.", config_name="examples/bob_orchestra_noReflection'''
    # config_name = "examples/bob_orchestra_noReflection"
    config_name = "examples/rl_train/qa_wiki_bob"
    agent_config = ConfigLoader.load_agent_config(config_name)
    # import pdb;pdb.set_trace();
    agent_config.max_turns = 30
    agent = get_agent(agent_config)

    await agent.build()
    try:
        recorder = await agent.run(query)
    finally:
        await agent.cleanup()

    # Most useful fields
    print("Query:\n", query)
    print("Answer:\n", recorder.final_output)
    return



if __name__ == "__main__":
    
    asyncio.run(main())

