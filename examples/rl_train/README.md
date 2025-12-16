## Setup
Setup Youtu-Agent following instructions in [GitHub](https://github.com/TencentCloudADP/youtu-agent)

The only environment variables needed here are LLM settings:

```sh
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=
UTU_LLM_BASE_URL=
UTU_LLM_API_KEY=
```


## Case 1: QA with Python tool

- config: `configs/agents/examples/rl_train/qa_python.yaml`
- tools: Python executor. Here we choose the built-in [Python tool](../../utu/tools/python_executor_toolkit.py) (which runs in a subprocess). You can also use remote code sandbox like [SandboxFusion](https://bytedance.github.io/SandboxFusion/).

```sh
python scripts/cli_chat.py --config examples/rl_train/qa_python
```


## Case 2: multi-hop QA with retrieval tool (wiki)

- config: `configs/agents/examples/rl_train/qa_wiki.yaml`
- retrieval API: serving the `Wikipedia 2018` dataset. You can use this [script](https://github.com/inclusionAI/ASearcher/blob/main/scripts/launch_local_server.sh) from ASearcher to launch a local wikipedia retrieval service. 
- tools: [wiki_tool.py](./wiki_tool.py). Note to set the correct retrieval API URL in the code.

```sh
python scripts/cli_chat.py --config examples/rl_train/qa_wiki
```

## Case 3: browser use

- config: `configs/agents/examples/rl_train/browser.yaml`
- environment: [BrowserTioneEnv](../../utu/env/browser_tione_env.py). Note that you should config the remote tione environment service first, see [iwiki](https://iwiki.woa.com/p/4015475657) and [TioneEnvManager](../../utu/env/utils/tione_manager.py).

```sh
python scripts/cli_chat.py --config_name examples/rl_train/browser
```

