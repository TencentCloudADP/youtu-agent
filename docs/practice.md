# Agent Practice with Training-Free GRPO

<a href=https://arxiv.org/abs/2510.08191><img src=https://img.shields.io/badge/arXiv-2510.08191-b31b1b.svg></a>

This guide covers the agent practice functionality in Youtu-Agent, powered by Training-Free Group Relative Policy Optimization (GRPO). Training-Free GRPO is a cost-effective solution that enhances agent performance without LLM parameter updates by leveraging group relative semantic advantages and iteratively distilling high-quality experiential knowledge.

## Overview

The practice module provides core functionality for:

- **Training-Free Learning**: Improve agent performance without fine-tuning model parameters
- **Experience Distillation**: Extract and integrate high-quality experiential knowledge
- **Flexible Evaluation**: Configurable reward calculation through custom verification functions
- **Domain Adaptation**: Support for diverse tasks from math reasoning to web search

## Module Structure

```
utu/practice/
├── __init__.py                 # Module exports
├── training_free_grpo.py       # Main orchestrator
├── rollout_manager.py          # Rollout execution and batch processing
├── experience_updater.py       # Experience processing and integration
├── data_manager.py             # Dataset management
├── utils.py                    # Configuration parsing and utilities
├── dataset/                    # Dataset storage directory
└── verify/                     # Verification functions
    ├── math.py                 # Math verification
    └── webwalker.py            # Web search verification
```

## Quick Start

### Prerequisites

Before starting, ensure you have:

1. Completed the [QuickStart](quickstart.md) guide for environment setup
2. Installed all dependencies: `uv sync --all-extras`
3. Activated the virtual environment: `source .venv/bin/activate`
4. Configured API keys in `.env` file

### Basic Workflow

The practice process follows these steps:

1. **Data Preparation**: Upload datasets for practice and evaluation
2. **Verification Setup**: Configure domain-specific verification functions
3. **Configuration**: Prepare agent, evaluation, and practice configs
4. **Baseline Evaluation**: Evaluate initial agent performance
5. **Run Training-Free GRPO**: Execute the practice process
6. **Evaluate Enhanced Agent**: Assess improved performance

## Configuration System

The practice module uses a hierarchical configuration approach:

### Configuration Hierarchy

```yaml
configs/
├── agents/practice/              # Agent configurations
│   ├── math_agent.yaml
│   ├── math_practice_agent.yaml
│   ├── web_agent.yaml
│   └── web_practice_agent.yaml
├── eval/                         # Evaluation configurations
│   ├── math/
│   │   ├── math_AIME24.yaml
│   │   └── math_AIME25.yaml
│   └── web/
│       ├── web.yaml
│       └── web_practice.yaml
└── practice/                     # Practice configurations
    ├── math_reasoning.yaml
    └── web_search.yaml
```

### Configuration Components

**TrainingFreeGRPOConfig**: Unified configuration class with:

- `exp_id`: Experiment identifier
- `PracticeArguments`: Practice-specific parameters (epochs, batch size, GRPO settings)
- `DataArguments`: Data processing parameters
- `EvalConfig`: Evaluation configuration reference

**Utilities**:

- `TaskRecorder`: Records practice progress, experiences, and statistics
- `parse_training_free_grpo_config()`: Configuration parser with YAML files and command-line overrides

## Data Preparation

### Upload from HuggingFace

Use the provided script to load built-in datasets:

```bash
python scripts/data/process_training_free_GRPO_data.py
```

Built-in datasets include:

- **AIME24/AIME25**: AIME competition problems
- **DAPO-Math-17k**: Math problems from DAPO dataset
- **AFM_web_RL**: Web agent reinforcement learning dataset
- **WebWalkerQA**: Web navigation question-answering dataset

### Upload Custom Datasets

Upload your own datasets from local files:

```bash
python scripts/data/upload_dataset.py \
  --file_path path/to/your_dataset.jsonl \
  --dataset_name YourDataset
```

**Required fields** for each sample:

```python
{
    "dataset": "YourDataset",           # Dataset name
    "source": "training_free_grpo",     # Must be "training_free_grpo"
    "question": "What is 2+2?",         # The question/prompt
    "answer": "4"                       # Expected answer (or None)
}
```

## Verification Functions

Verification functions are the core of the reward calculation system, providing domain-specific evaluation criteria.

### Function Interface

Create verification functions in `utu/practice/verify/`:

```python
from utu.db import EvaluationSample

def verify_func(sample: EvaluationSample, timeout_score: float = 0, **kwargs) -> dict:
    """
    Verify the correctness of an agent response.

    Args:
        sample: EvaluationSample containing:
            - raw_question: Original question
            - correct_answer: Ground truth answer
            - response: Agent's final response
            - other metadata fields
        timeout_score: Score for timeout cases
        **kwargs: Additional arguments including:
            - llm: LLM client for verification requiring judgment

    Returns:
        dict: {
          "reward": float,          # ranges from 0.0 to 1.0
          "reasoning": str | None   # extra details for experience extraction
        }
    """
    # Your verification logic here
    pass
```

### Built-in Verification Functions

**Math Verification** (`utu/practice/verify/math.py`):

- Uses symbolic math verification
- Compares extracted expressions with ground truth
- Requires `math-verify` package: `uv pip install math-verify`

**Web Search Verification** (`utu/practice/verify/webwalker.py`):

- LLM-based judgment for web search responses
- Compares agent response with ground truth using judge LLM
- Access judge via `kwargs['llm']`

### Custom Verification

Example for simple string matching:

```python
# utu/practice/verify/str_match.py
from utu.db import EvaluationSample

def string_match_verify(sample: EvaluationSample, timeout_score: float = 0, **kwargs) -> dict:
    """Simple string matching verification."""
    if sample.correct_answer.lower() == sample.response.lower():
        return {"reward": 1.0, "reasoning": None}
    return {"reward": 0.0, "reasoning": None}
```

## Configuration Files

### Agent Configuration

Create or use existing agent configs in `configs/agents/practice/`. See [Agents](agents.md) for detailed configuration options.

### Evaluation Configuration

Create evaluation config in `configs/eval/`:

```yaml
# configs/eval/my_domain/my_eval.yaml
# @package _global_
defaults:
  - /agents/practice/my_agent@agent
  - _self_

exp_id: "my_eval"

# Evaluation dataset
data:
  dataset: "MyEvalDataset"
  type: "single"

# Evaluation settings
concurrency: 64
pass_k: 3

# Verification function
verify_filename: "my_verify.py"
verify_func_name: "my_verify_func"

# Optional: Judge model for LLM-based verification
judge_model:
  model_provider:
    type: ${oc.env:JUDGE_LLM_TYPE}
    model: ${oc.env:JUDGE_LLM_MODEL}
    base_url: ${oc.env:JUDGE_LLM_BASE_URL}
    api_key: ${oc.env:JUDGE_LLM_API_KEY}
  model_params:
    temperature: 0.5
```

### Practice Configuration

Create practice config in `configs/practice/`:

```yaml
# configs/practice/my_practice.yaml
# @package _global_
defaults:
  - /eval/my_domain/my_eval@evaluation
  - _self_

exp_id: "my_practice"

# Practice Arguments
practice:
  epochs: 5
  batch_size: 32
  grpo_n: 3
  rollout_concurrency: 64
  rollout_temperature: 0.7
  task_timeout: 3600
  do_eval: false
  eval_strategy: "epoch"
  restart_step: null
  agent_objective: |
    input: Description of input
    output: Description of expected output
  learning_objective: |
    Description of learning goals and expected experiences
  num_experiences_per_query: 1

# Data Arguments
data:
  practice_dataset_name: "MyPracticeDataset"
```

## Running Practice

### Evaluate Baseline

First, evaluate the baseline agent:

```bash
python scripts/run_eval.py \
  --config_name my_domain/my_eval
```

### Execute Training-Free GRPO

Run the practice process:

```bash
# Using configuration file
python scripts/run_training_free_GRPO.py \
  --config_name my_practice

# With parameter overrides
python scripts/run_training_free_GRPO.py \
  --config_name my_practice \
  --experiment_name my_practice \
  --epochs 5 \
  --batch_size 64
```

### Restart Behavior

Control caching and restart with `--restart_step`:

```bash
# Complete restart (no caching)
python scripts/run_training_free_GRPO.py \
  --config_name my_practice \
  --restart_step 0

# Resume from cached results (default)
python scripts/run_training_free_GRPO.py \
  --config_name my_practice \
  --restart_step null

# Partial restart: cache steps 0-2, restart from step 3
python scripts/run_training_free_GRPO.py \
  --config_name my_practice \
  --restart_step 3
```

### Practice Output

The practice process generates:

1. **Enhanced Agent Configuration**: YAML file with integrated experiences
2. **Tracing Logs**: Detailed logs via Phoenix (if enabled):
   - Rollout trajectories
   - Experience extraction steps
   - Statistics at each step
   - Evaluation performance (if `do_eval` enabled)
3. **Experience Records**: Structured records in database

### Evaluate Enhanced Agent

After practice completes, evaluate the enhanced agent:

```bash
python scripts/run_eval.py \
  --config_name my_domain/my_practice
```

## Example Workflows

### Math Reasoning

Complete workflow for math reasoning tasks:

```bash
# Install dependencies
uv pip install math-verify

# Prepare data
python scripts/data/process_training_free_GRPO_data.py

# Evaluate baseline
python scripts/run_eval.py --config_name math/math_AIME24
python scripts/run_eval.py --config_name math/math_AIME25

# Run practice
python scripts/run_training_free_GRPO.py --config_name math_reasoning

# Evaluate enhanced agent
python scripts/run_eval.py --config_name math/math_practice_AIME24
python scripts/run_eval.py --config_name math/math_practice_AIME25
```

### Web Searching

Complete workflow for web search tasks:

```bash
# Setup environment variables in .env
SERPER_API_KEY=your-serper-api-key
JINA_API_KEY=your-jina-api-key

# Prepare data
python scripts/data/process_training_free_GRPO_data.py

# Evaluate baseline
python scripts/run_eval.py --config_name web/web

# Run practice
python scripts/run_training_free_GRPO.py --config_name web_search

# Evaluate enhanced agent
python scripts/run_eval.py --config_name web/web_practice
```

## Tracing & Monitoring

Enable Phoenix tracing for detailed monitoring:

```bash
# Install Phoenix
pip install arize-phoenix

# Start Phoenix server
phoenix serve

# Configure in .env
PHOENIX_ENDPOINT=http://127.0.0.1:6006/v1/traces
PHOENIX_PROJECT_NAME=Youtu-Agent
```

Phoenix provides visibility into:

- Rollout trajectories and agent decisions
- Experience extraction process
- Practice progress and statistics
- Evaluation metrics over time

## Advanced Topics

### Custom Reward Functions

For complex domains, you can create sophisticated verification functions that:

- Combine multiple evaluation criteria
- Use LLM judges for nuanced assessment
- Implement domain-specific metrics
- Provide detailed reasoning for experience extraction

### Multi-Stage Practice

For iterative improvement, you can:

1. Run initial practice on simpler datasets
2. Evaluate on progressively harder benchmarks
3. Continue practice with harder examples
4. Use `restart_step` to build on previous results

### Hyperparameter Tuning

Key parameters to optimize:

- `batch_size`: Samples per batch (affects memory and speed)
- `grpo_n`: Rollouts per group (higher = better but slower)
- `rollout_temperature`: LLM temperature during rollouts
- `num_experiences_per_query`: Experiences extracted per query

## API Reference

For detailed API documentation, see:

- [Evaluation Data](ref/eval/data.md)
- [Evaluation Processor](ref/eval/processor.md)
- [Benchmarks](ref/eval/benchmarks.md)

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{training_free_grpo,
      title={Training-Free Group Relative Policy Optimization},
      author={Tencent Youtu Lab},
      year={2025},
      eprint={2510.08191},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.08191},
}

@misc{youtu-agent-2025,
  title={Youtu-agent: A Simple yet Powerful Agent Framework},
  author={Tencent Youtu Lab},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TencentCloudADP/youtu-agent}},
}
```
