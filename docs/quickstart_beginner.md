# Beginner's Quickstart Guide 🚀

This is a detailed guide designed for beginners to walk through the setup and usage of the Youtu-agent framework. Even if you're new to AI Agent development, you'll be able to successfully run Youtu-agent.

## 📖 Prerequisites

Before getting started, please ensure your system meets the following requirements:

- **Python** 3.12 or higher.
- **[Git tool](https://git-scm.com/downloads)**.
- **`UV`** Package Manager: an extremely fast Python package and project manager. We'll install this in the setup steps below.
- **API Keys**: You'll need to obtain API keys for the underlying LLM that your agent will use (e.g., `DeepSeek`, `OpenAI`, etc.) - this is required. Optionally, you can also get API keys for `Serper` and `Jina` for enhanced features.
    - **`DeepSeek API Key`**: Visit [DeepSeek](https://platform.deepseek.com/) or [Tencent Cloud](https://cloud.tencent.com/document/product/1772/115969) and register an account to get an API key.
    - **`Serper API Key`** (optional): Visit [Serper](https://serper.dev/) and get API key.
    - **`Jina API Key`** (optional): Visit [Jina](https://jina.ai/reader/) and get API key.

---

## 🔧 Detailed Installation Steps

### Step 1: Clone the Project Code

Open a terminal (command line) and execute the following commands:

```sh
# Clone the project locally
git clone https://github.com/TencentCloudADP/youtu-agent.git

# Enter the project directory
cd youtu-agent
```

**Beginner Tip:** If you see `git: command not found`, it means Git is not installed. Please install Git first.

### Step 2: Install UV Package Manager

You can **install** `UV` using the following commands (refer to `UV` official repo's [installation guides](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)):

```sh
# Install UV on Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or simply use pip:

```sh
pip install uv
```

```sh
# Or pipx
pipx install uv
```

**Verify Installation:**

```bash
uv --version
```

If a version number is displayed, the installation was successful.

### Step 3: Create and Activate Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate
```

**Beginner Tip:** After activating the virtual environment, you'll see `(Youtu-agent)` identifier before your command line prompt as follow:

```sh
# Linux/macOS
(Youtu-agent) your-username:~/path/to/youtu-agent$

# Windows
# (Youtu-agent) path\to\Youtu-agent>
```

### Step 4: Install Project Dependencies

```sh
# Install all dependencies, including development tools
uv sync --group dev
```

### Step 5: Configure Environment Variables

```sh
# Copy environment variable template file
cp .env.example .env
```

Now you need to edit the `.env` file and add your API keys:

```sh
# Use a text editor to open the configuration file
# You can use nano, vim, or any editor you prefer
nano .env
```

In the opened file, find the following lines and fill in your API keys:

```bash
# LLM Configuration - **Required**
# We use DeepSeek as an example LLM provider.
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=[DeepSeek_API_key]
# Or use the DeepSeek equivalent on Tencent Cloud
# UTU_LLM_TYPE=chat.completions
# UTU_LLM_MODEL=deepseek-v3
# UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
# UTU_LLM_API_KEY=[DeepSeek_API_key]

# Tools Configuration - Optional
SERPER_API_KEY=[Serper_API_key]
JINA_API_KEY=[Jina_API_key]
```

More advanced configurations are available in the later [Advanced Setup](#-advanced-setup) section and [Configuration Documentation](./config.md).

**Important Reminder:** 
- Replace `[DeepSeek_API_key]` with your actual DeepSeek API Key
- If you don't have `Serper` and `Jina` API Keys yet, you can leave them empty, but some features may not work

---

## 🎯 First Run Tests

Let's verify that the installation was successful:

### Test 1: Run Simple Search Agent

```sh
# Run a simple agent with search capabilities as startup test
# python scripts/cli_chat.py --help
python scripts/cli_chat.py --config_name simple/search_agent.yaml
```

If everything is working correctly, you should see:

```
__   __            _                                      _   
\ \ / / ___  _  _ | |_  _  _  ___  __ _  __ _  ___  _ _  | |_ 
 \ V / / _ \| || ||  _|| || ||___|/ _` |/ _` |/ -_)| ' \ |  _|
  |_|  \___/ \_,_| \__| \_,_|     \__,_|\__, |\___||_||_| \__|
                                        |___/                 

----------------------------------------------------------------------------------------------------
Usage: python cli_chat.py --config_name <config_name>
Quit: exit, quit, q
----------------------------------------------------------------------------------------------------
>
```

Now you can try asking some questions:

```
> What can you do?
```

**Beginner Tip:** 
- Type `quit`, `exit` or `q` to exit the conversation
- If you encounter errors, check if your `UV` environment is activated and `UTU_LLM_*` API Key is configured correctly

### Test 2: Run Orchestra Example

Run a multi-agent (Plan-and-Execute) orchestra agent by specifying its configuration file:

```sh
# Run SVG generator example
python examples/svg_generator/main.py
```

This will start an agent that can generate SVG graphics code on your terminal (command line).

You can also run a web UI for the agent:

```sh
python examples/svg_generator/main_web.py
```

See more in [frontend](./frontend.md).

---

## Run Evaluations

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

## 🔧 Advanced Setup

Once you're comfortable with the basics, you might want to customize your setup further:

### Database Configuration

The evaluation framework uses a SQL database (defaulting to SQLite) to store datasets and experiment results. The default SQLite database (`sqlite:///test.db`) is perfect for getting started, but you can use other databases for production use.

To use a different database (e.g., PostgreSQL), set the `UTU_DB_URL` environment variable in your `.env` file:

```sh
# For PostgreSQL
UTU_DB_URL="postgresql://user:password@host:port/database"

# For MySQL
UTU_DB_URL="mysql://user:password@host:port/database"

# Default SQLite (recommended for beginners)
UTU_DB_URL="sqlite:///test.db"
```

**Beginner Tip:** Stick with SQLite unless you have specific requirements for a different database system.

### Tracing

We use [Phoenix](https://arize.com/docs/phoenix) as our default tracing service for observing agent behavior. This helps you understand what your agents are doing step-by-step.

To enable tracing, add these environment variables to your `.env` file:

```sh
# Phoenix Tracing Configuration
PHOENIX_ENDPOINT=[Phoenix_endpoint]
PHOENIX_BASE_URL=[Phoenix_base_url]
PHOENIX_PROJECT_NAME=[Phoenix_project_name]
```

The framework also supports any tracing service compatible with the `openai-agents` library. See the [official list of tracing processors](https://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list) for more options.

**Beginner Tip:** Tracing is optional but very helpful for debugging and understanding your agent's behavior. You can skip this initially and add it later when you want to dive deeper.

### Using Different LLM Providers

While our examples use DeepSeek, you can easily switch to other LLM providers by modifying your `.env` file:

```bash
# For OpenAI GPT models
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=gpt-4
UTU_LLM_BASE_URL=https://api.openai.com/v1
UTU_LLM_API_KEY=[Openai_API_key]

# For Anthropic Claude (via OpenAI-compatible API)
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=claude-3-sonnet-20240229
UTU_LLM_BASE_URL=[Anthropic_compatible_endpoint]
UTU_LLM_API_KEY=[Anthropic_API_key]
```

**Beginner Tip:** Start with one LLM provider and get familiar with the framework before experimenting with others.

---

## 🎨 Create Your First Custom Agent

Now that you've successfully run the project, let's create your own agent!

### Step 1: Create Configuration File

```sh
# Create a new agent configuration file
mkdir -p configs/agents/my_agents
```

Create file `configs/agents/my_agents/my_first_agent.yaml`:

```yaml
# @package _global_
defaults:
  - /model/base@model.   # Loads base LLM model settings
  - /tools/search@toolkits.search.  # Loads the builtin 'search' toolkit
  - _self_   # Loads the current configuration file, allowing overrides

agent:
    name: MyFirstAgent
    instructions: "You are a helpful assistant that can search the web."
```

### Step 2: Write and run the Python script

```python
import asyncio
from utu.agents import SimpleAgent

async def main():
    # Use your custom agent configuration
    async with SimpleAgent(config="my_agents/my_first_agent.yaml") as agent:
        # Ask a question
        await agent.chat("What's the weather in Beijing today?")

asyncio.run(main())
```

### Step 3: Test Functionality

Try asking some other questions in the script, such as:
- "Hello, please introduce yourself"
- "Please search for the latest tech news"
- "Help me analyze the development trends of artificial intelligence"

---

## 📚 Next Steps

Congratulations! You've successfully run Youtu-agent. Next, you can:

1. **Learn Configuration:** Read the complete [Configuration Documentation](./config.md) to learn how to customize agents and understand all available configuration options
2. **Add Tools:** Read the [Tools Documentation](./tools.md) to learn how to add new features to agents
3. **Explore More Examples:** Check various examples in [Examples](./examples.md) for more detailed use cases and advanced scripts
4. **Dive into Evaluations:** Learn more about how the evaluation framework works by reading the [Evaluation Framework documentation](./eval.md).

---

## 🆘 Getting Help

If you encounter issues while using the project, you can:

1. Check the [Complete Documentation](https://tencentcloudadp.github.io/youtu-agent/)
2. Ask questions in [GitHub Issues](https://github.com/TencentCloudADP/youtu-agent/issues)
3. View the project's [Example Code](https://github.com/TencentCloudADP/youtu-agent/tree/main/examples)

Happy coding! 🎉
