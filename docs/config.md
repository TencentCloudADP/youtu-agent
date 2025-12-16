# Configuration System

The project's configuration is managed through a system based on `pydantic` for data validation and `hydra` for loading from YAML files. This provides a powerful and flexible way to define agents and experiments.

All configurations are stored as `.yaml` files inside the `/configs` directory.

## `ConfigLoader`

The `ConfigLoader` is the main entry point for loading configurations from YAML files into `pydantic` models. It abstracts away the file paths and loading logic.

**Usage:**

```python
from utu.config import ConfigLoader

# Load an agent configuration from /configs/agents/my_agent.yaml
agent_config = ConfigLoader.load_agent_config("my_agent")

# Load an evaluation configuration from /configs/eval/my_eval.yaml
eval_config = ConfigLoader.load_eval_config("my_eval")
```

---

## `AgentConfig`

`AgentConfig` is the central data structure for defining an agent. It specifies everything the agent needs to operate, including its model, tools, and personality.

### Key Components

- **`type`**: The agent's architecture. Can be:
    - `simple`: A single agent that performs a task.
    - `orchestra`: A more complex, multi-agent system with a planner and workers.
- **`model`**: (`ModelConfigs`) Defines the primary LLM the agent will use, including the API provider, model name, and settings like temperature.
- **`agent`**: (`ProfileConfig`) Defines the agent's profile, such as its `name` and system-level `instructions` (e.g., "You are a helpful assistant.").
- **`env`**: (`EnvConfig`) Specifies the environment the agent operates in (e.g., `shell_local` or `browser_docker`). See [Agent Environments](./env.md) for more details.
- **`toolkits`**: (`dict[str, ToolkitConfig]`) A dictionary defining the tools available to the agent. Each toolkit can be loaded in `builtin` mode (running in the main process) or `mcp` mode (running as a separate process).
- **`max_turns`**: The maximum number of conversational turns the agent can take before stopping.

For the `orchestra` type, `AgentConfig` also includes fields for defining the planner, workers, and reporter agents.

---

## `EvalConfig`

`EvalConfig` defines a complete evaluation experiment. It specifies the dataset to use, the agent to test, and how to judge the results.

### Key Components for Evaluation

- **`data`**: (`DataConfig`) Defines the dataset to be used for the evaluation, including its name/path and the relevant fields (`question_field`, `gt_field`).
- **`rollout`**: This section defines the execution phase of the evaluation.
    - **`agent`**: (`AgentConfig`) A full `AgentConfig` for the agent being tested.
    - **`concurrency`**: The number of parallel processes to use when running the agent on the dataset.
- **`judgement`**: This section defines the judgment phase.
    - **`judge_model`**: (`ModelConfigs`) The configuration for the LLM that will act as the judge.
    - **`judge_concurrency`**: The number of parallel processes to use for judging the results.
    - **`eval_method`**: The method used for evaluation (e.g., comparing against a ground truth answer).
