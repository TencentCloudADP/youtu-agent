# Quickstart

This guide will walk you through setting up the project, running your first agent, and executing evaluations.

## Installation & Setup

First, clone the repository and set up the Python environment.

```sh
# Clone the project repository
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent

# We use `uv` to manage the virtual environment and dependencies
# Create the virtual environment
uv venv

# Activate the environment
source .venv/bin/activate

# Install all dependencies, including development tools
uv sync --group dev

# Create your environment configuration file from the example
cp .env.example .env
```

After creating the `.env` file, you **must** edit it to add your necessary API keys (e.g., `UTU_LLM_API_KEY`, `SERPER_API_KEY`, etc.).

---

## Running an Agent

You can interact with agents directly from the command line using the `cli_chat.py` script.

### Simple Agent

Run a simple agent defined by a configuration file. For example, to run an agent with search capabilities:

```sh
# python scripts/cli_chat.py --help
python scripts/cli_chat.py --config_name simple/search_agent.yaml
```

### Orchestra Agent

Run a multi-agent (Plan-and-Execute) orchestra agent by specifying its configuration file:

```sh
python examples/svg_generator/main.py
```

You can also run a web UI for the agent:

```sh
python examples/svg_generator/main_web.py
```

See more in [frontend](./frontend.md).

---

## Running Evaluations

The framework includes a powerful evaluation harness to benchmark agent performance.

### Run a Full Experiment

This command runs a complete evaluation, from agent rollout to judging.

```sh
python scripts/run_eval.py --config_name <your_eval_config> --exp_id <your_exp_id> --dataset WebWalkerQA --concurrency 5
```

### Re-judge Existing Results

If you have already run the rollout and only want to re-run the judgment phase, use this script:

```sh
python scripts/run_eval.py --config_name <your_eval_config> --exp_id <your_exp_id> --dataset WebWalkerQA --step judge
```

### Dump Experiment Data

You can also dump the trajectories and results from the database for a specific experiment:

```sh
python scripts/db/dump_db.py --exp_id "<your_exp_id>"
```

---

## Advanced Setup

### Database Configuration

The evaluation framework uses a SQL database (defaulting to SQLite) to store datasets and experiment results. To use a different database (e.g., PostgreSQL), set the `UTU_DB_URL` environment variable:

```sh
UTU_DB_URL="postgresql://user:password@host:port/database"
```

### Tracing

We use [Phoenix](https://arize.com/docs/phoenix) as our default tracing service for observing agent behavior. To enable it, set the following environment variables:
- `PHOENIX_ENDPOINT`
- `PHOENIX_BASE_URL`
- `PHOENIX_PROJECT_NAME`

The framework also supports any tracing service compatible with the `openai-agents` library. See the [official list of tracing processors](https://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list) for more options.

---


## Customizing the Agent

### Create a config file

```yaml
# configs/agents/sample_tool.yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search # Loads the 'search' toolkit
  - _self_

agent:
    name: simple-tool-agent
    instructions: "You are a helpful assistant that can search the web."
```

### Write and run the Python script

```python
import asyncio
from utu.agents import SimpleAgent

async def main():
    async with SimpleAgent(config="sample_tool.yaml") as agent:
        await agent.chat("What's the weather in Beijing today?")

asyncio.run(main())
```

## Next Steps

- **Explore Examples**: Check the `/examples` directory for more detailed use cases and advanced scripts.
- **Dive into Evaluations**: Learn more about how the evaluation framework works by reading the [Evaluation Framework documentation](./eval.md).
