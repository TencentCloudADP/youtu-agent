

# Training Youtu-Agent with Ease: Hands-On Guide for End-to-End Reinforcement Learning

<img src="docs/assets/youtu-agl-mascot.png" alt="Youtu-Agent x Agent Lightning logo" width="200" align="left" style="margin-right:20px;">

This repository allows you to train your agents built by Youtu-Agent. We have **verified the performace** of code/math (ReTool) and search (SearchR1) tasks with multi-node training on **128 GPUs**.

[**Youtu-Agent**](https://github.com/TencentCloudADP/youtu-agent/tree/rl/agl) is a framework for building and managing your own Youtu Agent. It is designed to be used either as a command-line tool or a library in your own Python projects.

[**Agent Lightning**](https://github.com/microsoft/agent-lightning/tree/contrib/youtu-agent-lightning) is a framework for training LLM-based agents via popular training backends. In the present implementation, we use the VeRL library for RL training.


## 1. Verified Training Performance

The RL training dynamics (at least 200 steps) of 7B instruct models are provided below for reference, confirming the effectiveness and stability of training with our repository. More performance gains are expected with prolonged training.


- [ReTool](https://api.wandb.ai/links/1275747829-fudan-university/vwxn21w2). AIME24: 0.10 (step 0) -> 0.45 (step 460).

<table>
  <tr>
    <td align="center"><img src="docs/assets/images/retool_entropy.png" width="200"/></td>
    <td align="center"><img src="docs/assets/images/retool_gradnorm.png" width="200"/></td>
    <td align="center"><img src="docs/assets/images/retool_val_ame24.png" width="200"/></td>
  </tr>
</table>


- [SearchR1](https://api.wandb.ai/links/yuleiqin-tencent/0e2hs7io). TriviaQA: 0.37 (step 0) -> 0.54 (step 200); PopQA: 0.16 (step 0) -> 0.35 (step 200); NQ: 0.24 (step 0) -> 0.45 (step 200); MuSiQue: 0.06 (step 0) -> 0.14 (step 200); HotpotQA: 0.21 (step 0) -> 0.38 (step 200); Bamboogle: 0.23 (step 0) -> 0.36 (step 200); 2wiki: 0.22 (step 0) -> 0.32 (step 200).

<table>
  <tr>
    <td align="center"><img src="docs/assets/images/search_entropy.png" width="200"/></td>
    <td align="center"><img src="docs/assets/images/search_gradnorm.png" width="200"/></td>
    <td align="center"><img src="docs/assets/images/search_val_triviaqa.png" width="200"/></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/assets/images/search_val_popqa.png" width="200"/></td>
    <td align="center"><img src="docs/assets/images/search_val_nq.png" width="200"/></td>
    <td align="center"><img src="docs/assets/images/search_val_musique.png" width="200"/></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/assets/images/search_val_hotpotqa.png" width="200"/></td>
    <td align="center"><img src="docs/assets/images/search_val_bamboogle.png" width="200"/></td>
    <td align="center"><img src="docs/assets/images/search_val_2wiki.png" width="200"/></td>
  </tr>
</table>


## 2. Installation

### 2.1 VeRL

To install VeRL, run the following command:

```bash
# create anaconda env (optional)
conda create -n agent-lightning python==3.12.0

# install verl
pip install verl==0.5
```

### 2.2 Agent Lightning

To install Agent Lightning, run the following command:

```bash
# install agent-lightning
git clone -b contrib/youtu-agent-lightning https://github.com/microsoft/agent-lightning.git
cd agent-lightning
pip install -e .
```

### 2.3 Youtu-Agent


To install Youtu-Agent, run the following command:

```bash
# install youtu-agent
git clone -b rl/agl https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
pip install -e .
# modify your .env accordingly
cp .env.example .env
```


## 3. Training Your Youtu-Agent

We provide two examples respectively for agents that: 1) solve Maths problems with codes (ReTool); 2) solve QA problems with local wiki search (SearchR1).

[ReTool](https://github.com/ReTool-RL/ReTool): We implement the agent via `configs/agents/retool/qa_python.yaml`. Its tool is **code interpreter** defined in `utu/tools/codesnip_toolkit.py`. Please make sure the local sandbox fusion service is ready and its IP address `server_url` is correctly set in the tool python file.

[SearchR1](https://github.com/PeterGriffinJin/Search-R1): We implement the agent via `configs/agents/examples/rl_train/qa_wiki.yaml`. Its tool is **local wiki search** defined in `examples/rl_train/wiki_tool.py`. Please make sure the local retrieval service is ready and its IP address `retrieval_service_url` is correctly set in the tool python file.



### 3.1 ReTool

For the detailed training and testing, please refer to this directory `examples_train_w_youtu/retool_youtu`.

1) For 7B model, we recommend at least 32 GPUs with 96GB memory.
2) Please modify the number of nodes and number of GPUs in `examples_train_w_youtu/retool_youtu/run_qwen2.5_7b_single_node.sh` and `examples_train_w_youtu/retool_youtu/run_qwen2.5_7b.sh` accordingly.
3) Make sure all the environment variables mentioned below are properly set.



#### Step 1

Download the training and testing datasets from the huggingface and save to `${PROJECT_DIR}/datasets` (e.g., datasets/BytedTsinghua-SIA/DAPO-Math-17k).

* Training Dataset 🤗 [https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k]
* Testing Dataset 🤗 [https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024]


#### Step 2

Download the SandboxFusion docker and launch the sandbox service:
* Sandbox Service ⌨️ [https://github.com/bytedance/SandboxFusion]


#### Step 3

Modify the training scripts and the agent config file to make sure every directory path and URL address is valid. For example:


- `examples_train_w_youtu/retool_youtu/run_qwen2.5_7b.sh`: BASE_MODEL, CODESNIP_SERVER_URL
- `examples_train_w_youtu/retool_youtu/sandbox_fusion_tool_config.yaml`: sandbox_fusion_url
- `utu/tools/codesnip_toolkit.py`: server_url (sandbox fusion service)


#### Step 4

Train the Youtu-Agent on a single node with 8 GPUs:

```bash
# restart the ray cluster
bash scripts/restart_ray.sh
# submit the ray training job
bash examples_train_w_youtu/retool_youtu/run_qwen2.5_7b_single_node.sh
```

(Optional) Train the Youtu-Agent on four nodes with 32 GPUs:

```bash
# submit the ray training job with the multi-node ray script
bash run_ray.sh examples_train_w_youtu/retool_youtu/run_qwen2.5_7b.sh
```

(Optional) Debugging and Testing

* Deployment of vLLM service

**Prerequisites:** Before starting the agent, please ensure that you have installed youtu-agent, and that the `retool` directory from `examples_train_w_youtu/retool_youtu/retool` is placed in `youtu-agent/configs/agents/retool`.

```bash
# launch vLLM backend server
export BASE_MODEL="YOUR_MODEL_PATH"
bash vllm_deploy.sh

# run the agent code
# You must launch the sandbox server first! (SandBoxFusion)
export CODESNIP_SERVER_URL="YOUR_SANDBOX_URL"
python calc_sandbox_agent_youtu.py
```

* Deployment of Store and Runner service
1. Store

```bash
agl store --port 9999
```

2. Runner

```bash
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=runner python train_calc_sandbox_agent.py --external-store-address http://localhost:9999 --n-runners 10
```


### 3.2 SearchR1

For the detailed training and testing, please refer to this directory `examples_train_w_youtu/search_r1_youtu`.

1) For 3B model, we recommend at least 2 GPUs with 96GB memory. For 32B model, we recommend at least 32 GPUs with 96GB memory.
2) Please modify the number of nodes and number of GPUs in the `examples_train_w_youtu/search_r1_youtu/train_search_agent.py`.
3) Make sure all the environment variables mentioned below are properly set.
4) It is noted that for reward score, we use both rule-based (exact-match) and llm-based (llm-as-a-judge) scoring techniques. Therefore, a llm service (openai-compatible) URL should be prepared in advance.



#### Step 1

Download the training and testing datasets from the huggingface and save to `${PROJECT_DIR}/datasets/asearcher_data/` (e.g., datasets/asearcher_data/ASearcher-train-data/base).

* Training Dataset 🤗 [https://huggingface.co/datasets/inclusionAI/ASearcher-train-data]
* Testing Dataset 🤗 [https://huggingface.co/datasets/inclusionAI/ASearcher-test-data]

Run the following script for data preprocessing:

```bash
bash examples_train_w_youtu/search_r1_youtu/data_preprocess/run_preprocess.sh
```

#### Step 2


Download the SandboxFusion docker and launch the sandbox service:
* Retrieval Service 🔍 [https://github.com/inclusionAI/ASearcher/blob/main/scripts/launch_local_server.sh]

#### Step 3

Modify the training scripts and the agent config file to make sure every directory path and URL address (retrieval service IP) is valid. For example:

- `examples_train_w_youtu/search_r1_youtu/search_tool_config.yaml`: YOUR_RETRIEVAL_SERVICE_IP
- `examples_train_w_youtu/search_r1_youtu/trainer7b_utu_onpolicy.sh`: MODEL_ROOT_PATH, REWARD_MODEL_URL, REWARD_MODEL_NAME
- `examples/rl_train/wiki_tool.py`: retrieval_service_url


#### Step 4


Train the Youtu-Agent on 4 nodes with 32 GPUs:
```bash
# 3B model
bash run_ray.sh examples_train_w_youtu/search_r1_youtu/trainer3b_utu_onpolicy.sh

# 32B model
bash run_ray.sh examples_train_w_youtu/search_r1_youtu/trainer32b_utu_onpolicy.sh
```


## Acknowledgement

We sincerely appreciate the efforts from the following projects:

* Youtu-Agent
```
@misc{youtu-agent-2025,
  title={Youtu-Agent: A Simple yet Powerful Agent Framework},
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

