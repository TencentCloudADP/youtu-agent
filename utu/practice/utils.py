import argparse
import pathlib
from dataclasses import dataclass
from typing import Any

from ..config import ConfigLoader, TrainingFreeGRPOConfig

CURR_DIR = pathlib.Path(__file__).parent
VERIFY_DIR = CURR_DIR / "verify"
DATASET_DIR = CURR_DIR / "dataset"


@dataclass
class TaskRecorder:
    """Record information about training-free GRPO process."""

    experiment_name: str = None
    """Name of the experiment"""
    experiences: dict[str, str] = None
    """Mapping from experience ID to experience content"""
    stats: dict[str, Any] = None
    """Mapping from stat name to stat value"""

    def experiences_update(self, experiences: dict[str, str]):
        self.experiences = experiences

    def stat_update(self, stat: dict[str, Any]):
        if self.stats is None:
            self.stats = {}
        self.stats.update(stat)


def parse_training_free_grpo_config() -> TrainingFreeGRPOConfig:
    """Parse a dictionary into TrainingFreeGRPOConfig."""
    parser = argparse.ArgumentParser(
        description="Run training-free GRPO experience generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration loading option
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        default="math_reasoning",
        help="Configuration name to load from configs/practice/ directory",
    )

    # Main configuration
    parser.add_argument("--experiment_name", type=str, default=None, help="name of experiment run")

    # Agent configurations
    parser.add_argument("--agent_config", type=str, default=None, help="Path to agent config YAML file")

    # Practice arguments
    practice_group = parser.add_argument_group("Practice Arguments")
    practice_group.add_argument("--epochs", type=int, default=None, help="number of practice epochs")
    practice_group.add_argument("--batch_size", type=int, default=None, help="Practice batch size")
    practice_group.add_argument("--grpo_n", type=int, default=None, help="Number of rollouts in a group of GRPO")
    practice_group.add_argument("--rollout_concurrency", type=int, default=None, help="Concurrency level for rollouts")
    practice_group.add_argument(
        "--rollout_data_truncate", type=int, default=None, help="Truncate data to first N samples"
    )
    practice_group.add_argument(
        "--eval_strategy", type=str, default=None, choices=["epoch", "steps"], help="Evaluation strategy"
    )
    practice_group.add_argument(
        "--eval_steps", type=int, default=None, help="Evaluation steps (when using 'steps' strategy)"
    )
    practice_group.add_argument(
        "--eval_data_truncate", type=int, default=None, help="Truncate evaluation data to first N samples"
    )
    practice_group.add_argument(
        "--restart_step",
        type=int,
        default=None,
        help="Step number to restart from (None means use cache for all steps if available, "
        "0 means restart from beginning)",
    )
    practice_group.add_argument(
        "--agent_objective",
        type=str,
        default=None,
        help="Clearly describe the objective of the working agent, briefly state its input and output.",
    )
    practice_group.add_argument(
        "--learning_objective",
        type=str,
        default=None,
        help="Clearly describe the learning direction and goal for the agent, as specific and detailed as possible.",
    )
    practice_group.add_argument(
        "--num_experiences_per_query",
        type=int,
        default=None,
        help="Number of experiences to generate per query during practice.",
    )

    # Data arguments
    data_group = parser.add_argument_group("Data Arguments")
    data_group.add_argument("--practice_dataset_name", type=str, default=None, help="Name of the practice dataset")

    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation Arguments")
    eval_group.add_argument(
        "--verify_filename", type=str, default=None, help="Path to Python file containing verify function"
    )
    eval_group.add_argument("--verify_func_name", type=str, default=None, help="Name of verify function in the file")
    eval_group.add_argument("--pass_k", type=int, default=None, help="Number of pass k for evaluation")

    args = parser.parse_args()

    # Load configuration
    config = ConfigLoader.load_training_free_grpo_config(args.config_name)
    if args.epochs is not None:
        config.practice.epochs = args.epochs
    if args.batch_size is not None:
        config.practice.batch_size = args.batch_size
    if args.grpo_n is not None:
        config.practice.grpo_n = args.grpo_n
    if args.rollout_concurrency is not None:
        config.practice.rollout_concurrency = args.rollout_concurrency
    if args.rollout_data_truncate is not None:
        config.practice.rollout_data_truncate = args.rollout_data_truncate
    if args.eval_strategy is not None:
        config.practice.eval_strategy = args.eval_strategy
    if args.eval_steps is not None:
        config.practice.eval_steps = args.eval_steps
    if args.eval_data_truncate is not None:
        config.practice.eval_data_truncate = args.eval_data_truncate
    if args.restart_step is not None:
        config.practice.restart_step = args.restart_step
    if args.agent_objective is not None:
        config.practice.agent_objective = args.agent_objective
    if args.learning_objective is not None:
        config.practice.learning_objective = args.learning_objective
    if args.num_experiences_per_query is not None:
        config.practice.num_experiences_per_query = args.num_experiences_per_query
    if args.practice_dataset_name is not None:
        config.data.practice_dataset_name = args.practice_dataset_name
    if args.experiment_name is not None:
        config.exp_id = args.experiment_name
    if args.agent_config is not None:
        agent_config = ConfigLoader.load_agent_config(args.agent_config)
        config.evaluation.agent = agent_config
    if args.verify_filename is not None:
        config.evaluation.verify_filename = args.verify_filename
    if args.verify_func_name is not None:
        config.evaluation.verify_func_name = args.verify_func_name
    if args.pass_k is not None:
        config.evaluation.pass_k = args.pass_k

    # Set evaluation exp_id to the same as the overall exp_id
    if config.exp_id is not None:
        config.evaluation.exp_id = config.exp_id
    return config
