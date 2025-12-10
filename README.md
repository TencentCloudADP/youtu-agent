

## Youtu-agent x Agent Lightning

### 1. Introduction

<img src="docs/assets/youtu-agl-mascot.png" alt="Youtu-agent x Agent Lightning logo" width="200" align="left" style="margin-right:20px;">


[**Youtu-agent**](https://github.com/TencentCloudADP/youtu-agent/tree/rl/agl) is a tool for building and managing your own Youtu agent. It is designed to be used either as a command-line tool or a library in your own Python projects.

[**Agent Lightning**](https://github.com/microsoft/agent-lightning/tree/contrib/youtu-agent-lightning) is a tool for training LLM-based agents with agent frameworks via popular training backends. In the present implementation, we use the VeRL library for RL training.


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

### 3. Training Your Youtu-agent

We provide two examples respectively for agents that: 1) solve maths problems with codes (ReTool); 2) solve qa problems with local wiki search (SearchR1).

[ReTool](https://github.com/ReTool-RL/ReTool): We implement the agent via `configs/agents/retool/qa_python.yaml`. Its tool is **code interpreter** defined in `utu/tools/codesnip_toolkit.py`. Please make sure the local sandbox fusion service is ready and its IP address `server_url` is correctly set in the tool python file.

[SearchR1](https://github.com/PeterGriffinJin/Search-R1): We implement the agent via `configs/agents/examples/rl_train/qa_wiki.yaml`. Its tool is **local wiki search** defined in `examples/rl_train/wiki_tool.py`. Please make sure the local retrieval service is ready and its IP address `retrieval_service_url` is correctly set in the tool python file.

The detailed usage of training Youtu-agent with Agent Lightning is provided in the [README](https://github.com/yuleiqin/youtu_agent_lightning/blob/youtu_agent_lightning/README.md).

#### 3.1 ReTool

1. Download the training and testing datasets from the huggingface and save to `${PROJECT_DIR}/datasets` (e.g., datasets/BytedTsinghua-SIA/DAPO-Math-17k).

* Training Dataset 🤗 [https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k]
* Testing Dataset 🤗 [https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024]

2. Download the SandboxFusion docker and launch the sandbox service:
* Sandbox Service ⌨️ [https://github.com/bytedance/SandboxFusion]

3. Modify the training scripts and the agent config file to make sure every directory path and URL address is valid. For example:

##### AgentLightning
- `examples_train_w_youtu/retool-youtu/run_qwen2.5_7b.sh`: BASE_MODEL, CODESNIP_SERVER_URL
- `examples_train_w_youtu/retool-youtu/sandbox_fusion_tool_config.yaml`: sandbox_fusion_url

##### Youtu-agent
- `utu/tools/codesnip_toolkit.py`: server_url (sandbox fusion service)


4. Train the Youtu-agent on a single node with 8 GPUs:
```
# restart the ray cluster
bash scripts/restart_ray.sh
# submit the ray training job
bash examples_train_w_youtu/retool-youtu/run_qwen2.5_7b_single_node.sh
```

(Optional) Train the Youtu-agent on four nodes with 32 GPUs:
```
# submit the ray training job with the multi-node ray script
bash run_ray.sh examples_train_w_youtu/retool-youtu/run_qwen2.5_7b.sh
```


##### 3.2 SearchR1

1. Download the training and testing datasets from the huggingface and save to `${PROJECT_DIR}/datasets/asearcher_data/` (e.g., datasets/asearcher_data/ASearcher-train-data/base).

* Training Dataset 🤗 [https://huggingface.co/datasets/inclusionAI/ASearcher-train-data]
* Testing Dataset 🤗 [https://huggingface.co/datasets/inclusionAI/ASearcher-test-data]

2. Download the SandboxFusion docker and launch the sandbox service:
* Retrieval Service 🔍 [https://github.com/inclusionAI/ASearcher/blob/main/scripts/launch_local_server.sh]

3. Modify the training scripts and the agent config file to make sure every directory path and URL address is valid. For example:

##### AgentLightning
- `examples_train_w_youtu/search_r1-youtu/search_tool_config.yaml`: YOUR_RETRIEVAL_SERVICE_IP
- `examples_train_w_youtu/search_r1-youtu/trainer7b_utu_onpolicy.sh`: MODEL_ROOT_PATH, REWARD_MODEL_URL, REWARD_MODEL_NAME (**We use both rule-based and LLM-as-a-Judge for rewards.**)

##### Youtu-agent
- `examples/rl_train/wiki_tool.py`: retrieval_service_url

4. Train the Youtu-agent on 4 nodes with 32 GPUs:
```
# 3B model
bash run_ray.sh examples_train_w_youtu/search_r1_youtu/trainer3b_utu_onpolicy.sh

# 32B model
bash run_ray.sh examples_train_w_youtu/search_r1_youtu/trainer32b_utu_onpolicy.sh
```

### Acknowledgement

We sincerely appreciate the efforts from the following projects:

* Youtu-agent
```
@misc{youtu-agent-2025,
  title={Youtu-agent: A Simple yet Powerful Agent Framework},
  author={Tencent Youtu Lab},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TencentCloudADP/youtu-agent}},
}
```


* AgentLightning
```
@misc{luo2025agentlightningtrainai,
      title={Agent Lightning: Train ANY AI Agents with Reinforcement Learning},
      author={Xufang Luo and Yuge Zhang and Zhiyuan He and Zilong Wang and Siyun Zhao and Dongsheng Li and Luna K. Qiu and Yuqing Yang},
      year={2025},
      eprint={2508.03680},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.03680},
}
```

* VeRL
```
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

