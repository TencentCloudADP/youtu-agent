

## Youtu-agent x Agent Lightning

### 1. Introduction

Youtu-agent is a tool for building and managing your own Youtu agent. It is designed to be used either as a command-line tool or a library in your own Python projects.

Agent Lightning is a tool for training LLM-based agents with agent frameworks via popular training backends. In the present implementation, we use the VeRL library for RL training.


### 2. Installation

#### 2.1 VeRL

To install VeRL, run the following command:

```bash
pip install verl==0.5
```

#### 2.2 Agent Lightning

To install Agent Lightning, run the following command:

```bash
pip install git+https://github.com/microsoft/agent-lightning.git@contrib/youtu-agent-lightning
```

#### 2.3 Youtu-agent


To install Youtu-agent, run the following command:

```bash
pip install git+https://github.com/TencentCloudADP/youtu-agent.git@rl/agl
```

### 3. Usage

We provide two examples respectively for agents that: 1) solve maths problems with codes (ReTool); 2) solve qa problems with local wiki search (SearchR1).

[ReTool](https://github.com/ReTool-RL/ReTool): We implement the agent via `configs/agents/retool/qa_python.yaml`. Its tool is **code interpreter** defined in `utu/tools/codesnip_toolkit.py`. Please make sure the local sandbox fusion service is ready and its IP address `server_url` is correctly set in the tool python file.

[SearchR1](https://github.com/PeterGriffinJin/Search-R1): We implement the agent via `configs/agents/examples/rl_train/qa_wiki.yaml`. Its tool is **local wiki search** defined in `examples/rl_train/wiki_tool.py`. Please make sure the local retrieval service is ready and its IP address `retrieval_service_url` is correctly set in the tool python file.

The detailed usage of training Youtu-agent with Agent Lightning is provided in the [README](https://github.com/yuleiqin/youtu_agent_lightning/blob/youtu_agent_lightning/README.md).

