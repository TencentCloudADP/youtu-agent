# Automatic Generation of Agents and Tools

A key feature of the `Youtu-Agent` framework is its ability to automate the creation of both tools and agent configurations. This streamlines the development process and reduces the need for manual boilerplate code and configuration.

## Automatic Tool Generation

### Overview

- Tool deployment: isolated environment; MCP-based communication.
- Generation approach: implement the tool's capability atomically, test it, then wrap it as an MCP tool.
- Integration details: a `manifest.json` specifies how the tool integrates with the `Youtu-Agent` framework.

### 1. Generate & Test the Tool

Run the following command to start the tool generation process:

```sh
python scripts/gen_tool.py
```

This script will create a new directory (`configs/tools/generated/{name}`) and configuration file (`configs/tools/generated/{name}.yaml`). It will also automatically create a virtual environment for the new tool and install its dependencies by running the following commands in the new directory:

```sh
cd {output_directory}
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Integrate the Tool into Your Agent

For instance, if the generated tool is named `download_bilibili_video`, you can add it to your agent configuration (`cofnigs/agents/bilibili.yaml`) as follows:

```yaml
# @package _global_
defaults:
  - /model/base@model
  - /tools/generated/download_bilibili_video@toolkits.download_bilibili_video
  - _self_

agent:
  name: utu-base
  instructions: "You are a helpful assistant."
```

Then, interact with your agent by running:

```bash
python scripts/cli_chat.py --config bilibili
```


## Automatic Agent Generation

`Youtu-Agent` can also automatically generate a configuration for a `SimpleAgent` based on your requirements. This is handled by an interactive "meta-agent" that asks you questions to define the agent's name, instructions, and desired tools.

<iframe width="100%" aspect-ratio="16/9" src="https://www.youtube.com/embed/JVpHDJtKBo8?si=tM7_PUEdhHZxMWlY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### 1. Generate the Agent Configuration

Start the interactive generation process by running:

```bash
python scripts/gen_simple_agent.py
```

The script will guide you through the setup process and save the resulting configuration file in the `configs/agents/generated/` directory.

### 2. Run the Generated Agent

Once the configuration is created, you can run your new agent using the `cli_chat.py` script. Be sure to replace `xxx` with the name of your generated config file.

```bash
python scripts/cli_chat.py --config generated/xxx
```
