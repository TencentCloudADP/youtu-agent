## Entry Point
```sh
python scripts/run_eval.py --config_name gaia --exp_id test_gaia_0906 --concurrency 5
```

## Remarks
- config: 
    - `configs/eval/gaia.yaml` for benchmark evaluation
    - `configs/agents/examples/gaia.yaml` main agent's config. You can config workforce configs and used subagents here.
    - `configs/agents/simple_agents/gaia_*.yaml` subagents' config. You can config prompts and used tools for each subagent.
- tools:
    - `examples/gaia/tools/browser_toolkit.py`: simply wrap [camel's HybridBrowserToolkit](https://github.com/camel-ai/camel/blob/master/camel/toolkits/hybrid_browser_toolkit_py/hybrid_browser_toolkit.py)
    - `examples/gaia/tools/search_toolkit.py`: customized search toolkit with `multi_query_deep_search, multi_query_parallel_search`, etc.
- agent implementation:
    - `utu/agents/workforce_agent.py`: main agent. The `WorkforceAgent` class implements a "plan-and-execute" agent with retry, replan capabilities.
