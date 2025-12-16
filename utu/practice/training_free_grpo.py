"""
Main module for experience generation. Control the process of Training-free GRPO.
"""

import os

import yaml
from agents import custom_span, function_span, gen_trace_id, trace

from ..config import TrainingFreeGRPOConfig
from ..config.eval_config import DataConfig
from ..utils import DIR_ROOT, get_logger
from ..utils.experience_cache import ExperienceCache
from .data_manager import TrainingFreeGRPODataManager
from .experience_updater import ExperienceUpdater
from .rollout_manager import RolloutManager
from .utils import TaskRecorder

logger = get_logger(__name__)


class TrainingFreeGRPO:
    config: TrainingFreeGRPOConfig = None
    practice_rollout_manager: RolloutManager = None
    eval_rollout_manager: RolloutManager = None
    experience_updater: ExperienceUpdater = None
    recorder: TaskRecorder = None

    def __init__(self, config: TrainingFreeGRPOConfig):
        """Initialize TrainingFreeGRPO with unified configuration."""
        self.config = config
        self.recorder: TaskRecorder = TaskRecorder(experiment_name=config.exp_id)

    async def run(self) -> str:
        """Run the complete experience generation process.

        Returns:
            str: Agent configuration file content in YAML format with experiences integrated
        """
        logger.info("Starting experience generation...")

        # Stage 0: Load components if not already built
        if self.practice_rollout_manager is None:
            logger.info("Stage 0: Building Training-free GRPO components...")
            await self.build()

        try:
            # Stage 1: Run training-free GRPO process
            logger.info("Stage 1: Running training-free GRPO process...")
            await self.practice()

            # Stage 2: Extract and process experiences
            logger.info("Stage 2: Extracting and processing experiences...")
            experiences = self.recorder.experiences or {}
            logger.info(f"Extracted {len(experiences)} experiences")
            agent_config_path = self._create_agent_config_with_experiences(experiences)
            return agent_config_path

        except Exception as e:
            logger.error(f"Error during experience generation: {e}", exc_info=True)
            raise

    async def build(self):
        """Build all components needed for training-free GRPO."""

        # 1. Load dataset
        # check if dataset exists
        data_manager = TrainingFreeGRPODataManager(self.config.evaluation)
        # load practice dataset if not exists
        if not data_manager.check_dataset(self.config.data.practice_dataset_name):
            raise ValueError(
                f"Practice dataset {self.config.data.practice_dataset_name} does not exist in db. Please load it first."
            )
        # load eval dataset if not exists
        if self.config.evaluation.data.dataset and not data_manager.check_dataset(self.config.evaluation.data.dataset):
            raise ValueError(
                f"Evaluation dataset {self.config.evaluation.data.dataset} does not exist in db. Please load it first."
            )

        # 2. Create practice rollout manager
        practice_eval_config = self.config.evaluation.model_copy()
        practice_eval_config.pass_k = self.config.practice.grpo_n
        self.original_temperature = practice_eval_config.agent.model.model_settings.temperature
        practice_eval_config.agent.model.model_settings.temperature = self.config.practice.rollout_temperature
        practice_eval_config.data = DataConfig(dataset=self.config.data.practice_dataset_name)
        self.practice_rollout_manager = RolloutManager(
            config=practice_eval_config, batch_size=self.config.practice.batch_size
        )

        # 3. Create eval rollout manager (if different from practice)
        self.eval_rollout_manager = None
        if self.config.practice.do_eval:
            eval_eval_config = self.config.evaluation.model_copy()
            eval_eval_config.exp_id = eval_eval_config.exp_id + "_eval"
            # eval_eval_config.data = DataConfig(dataset=self.config.data.eval_dataset_name)
            self.eval_rollout_manager = RolloutManager(
                config=eval_eval_config, batch_size=self.config.practice.batch_size
            )

        # 4. Create experience updater
        self.experience_updater = ExperienceUpdater(
            self.config.evaluation.agent, self.config.practice.agent_objective, self.config.practice.learning_objective
        )

        logger.info("Training-free GRPO components built successfully")

    async def practice(self):
        """Run practice process."""
        for epoch in range(self.config.practice.epochs):
            logger.info(f"Start Epoch {epoch}")

            # Prepare epoch data
            epoch_data = self.practice_rollout_manager.load_epoch_data(
                epoch, shuffle=self.config.practice.shuffle_data, truncate=self.config.practice.rollout_data_truncate
            )

            # check the batch size
            assert len(epoch_data) % self.config.practice.grpo_n == 0, (
                f"Epoch data size {len(epoch_data)} is not divisible by grpo_n {self.config.practice.grpo_n}"
            )
            if len(epoch_data) < self.config.practice.batch_size * self.config.practice.grpo_n:
                raise ValueError(
                    f"Epoch {epoch} data size {len(epoch_data) // self.config.practice.grpo_n} is smaller than "
                    f"batch size {self.config.practice.batch_size}."
                )
            if len(epoch_data) % (self.config.practice.batch_size * self.config.practice.grpo_n) != 0:
                logger.warning(
                    f"Epoch {epoch} data size {len(epoch_data) // self.config.practice.grpo_n} is not divisible by "
                    f"batch size {self.config.practice.batch_size}. Some data will be dropped."
                )

            # inner loop for each batch
            num_batches = len(epoch_data) // (self.config.practice.batch_size * self.config.practice.grpo_n)
            for batch_idx in range(num_batches):
                step = epoch * num_batches + batch_idx
                logger.info(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
                # set tracing
                step_trace_id = gen_trace_id()
                with trace(f"[{self.recorder.experiment_name}] Step {step} practice", trace_id=step_trace_id):
                    # get current stat
                    stats = self.recorder.stats or {}
                    if f"step_{step}" not in stats:
                        stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}

                    # 1. Rollout batch data
                    with custom_span("Process the batch data"):
                        rollouts, stat = await self.practice_rollout_manager.main(
                            batch_idx=batch_idx,
                            recorder=self.recorder,
                            use_cache=self._should_use_cache(step),
                        )
                        stats[f"step_{step}"]["rollout"] = stat

                    # 2. Update experiences based on rollouts
                    with custom_span("Generate batch experiences"):
                        # Check database cache first
                        cached_experiences = ExperienceCache.load_experiences(
                            experiment_name=self.recorder.experiment_name, step=step
                        )
                        if cached_experiences is not None and self._should_use_cache(step):
                            logger.info(
                                f"Experiences for step {step} already exist in database, skipping experience update."
                            )
                            new_experiences = cached_experiences
                            self.recorder.experiences_update(new_experiences)
                        else:
                            # If not cached, run experience updater
                            new_experiences = await self.experience_updater.run(
                                rollouts=rollouts,
                                recorder=self.recorder,
                                concurrency=self.config.practice.rollout_concurrency,
                                given_ground_truth=self.config.practice.given_ground_truth,
                                num_experiences=self.config.practice.num_experiences_per_query,
                            )

                            # Save to database cache
                            ExperienceCache.save_experiences(
                                experiment_name=self.recorder.experiment_name,
                                step=step,
                                experiences=new_experiences,
                                epoch=epoch,
                                batch=batch_idx,
                            )
                            logger.info(f"Step {step} completed. New experiences added: {len(new_experiences)}")

                        stats[f"step_{step}"]["complete"] = True
                        self.recorder.stat_update({f"step_{step}": stats[f"step_{step}"]})

                    # 3. Evaluation based on strategy
                    if self.eval_rollout_manager and self._should_evaluate(step, batch_idx, num_batches):
                        eval_trace_id = gen_trace_id()
                        with trace(f"[{self.recorder.experiment_name}] Step {step} evaluation", trace_id=eval_trace_id):
                            logger.info(f"Running evaluation at step {step}")
                            eval_data = self.eval_rollout_manager.load_epoch_data(
                                epoch=epoch, shuffle=False, truncate=self.config.practice.eval_data_truncate
                            )
                            logger.info(f"Evaluation dataset loaded with {len(eval_data)} records")
                            _, eval_stats = await self.eval_rollout_manager.main(
                                recorder=self.recorder, use_cache=self._should_use_cache(step)
                            )
                            with function_span("Record evaluation stats") as eval_stat_span:
                                eval_stat_span.span_data.output = eval_stats
                            stats[f"step_{step}"]["eval"] = eval_stats
                            self.recorder.stat_update({f"step_{step}": stats[f"step_{step}"]})

                    # 4. record stats and experiences to tracing
                    with function_span("Record current stats") as stat_span:
                        stat_span.span_data.output = stats[f"step_{step}"]
                    with function_span("Record current experiences") as exp_span:
                        exp_span.span_data.output = new_experiences

    def _should_use_cache(self, step: int) -> bool:
        """Determine if cached results should be used for current step.

        Restart behavior:
        - restart_step=None: Use cache for all steps (if available)
        - restart_step=N: Use cache for steps < N, execute fresh from step N onwards
        - restart_step=0: Execute all steps fresh (no caching)
        """
        restart_step = self.config.practice.restart_step
        return restart_step is None or step < restart_step

    def _should_evaluate(self, total_steps: int, batch_idx: int, num_batches: int) -> bool:
        """Determine if evaluation should be performed at current step."""
        if self.config.practice.eval_strategy == "epoch":
            # Evaluate at the end of each epoch
            return batch_idx == num_batches - 1
        elif self.config.practice.eval_strategy == "steps":
            # Evaluate every eval_steps
            return total_steps % self.config.practice.eval_steps == 0
        return False

    def _create_agent_config_with_experiences(self, experiences: dict[str, str]) -> str:
        """Create agent configuration with experiences integrated into instructions."""
        # Load the original agent config
        base_config = self.config.evaluation.agent
        # Convert to dict for manipulation
        config_dict = base_config.model_dump(exclude_none=True)

        # Format experiences for insertion
        if experiences:
            experience_text = "\n\nWhen solving problems, you MUST first carefully read and understand "
            experience_text += "the helpful instructions and experiences:\n"
            experience_text += "\n".join([f"[{i}]. {e}" for i, e in experiences.items()])
            # Insert experiences at the end of instructions
            current_instructions = config_dict.get("agent", {}).get("instructions", "You are a helpful assistant.")
            config_dict["agent"]["instructions"] = current_instructions + experience_text
            config_dict["model"]["model_settings"]["temperature"] = self.original_temperature

        # Remove unnecessary fields
        remain_default_keys = ["type", "model", "agent", "toolkits"]
        for key in list(config_dict.keys()):
            if key not in remain_default_keys:
                del config_dict[key]

        # Convert to YAML format
        yaml_config = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)
        config_header = "# @package _global_\ndefaults:\n  - _self_\n\n"
        # save to file
        config_filename = f"{self.config.evaluation.exp_id}_agent.yaml"
        config_dir = str(DIR_ROOT / "configs" / "agents" / "practice")
        full_path = os.path.join(config_dir, os.path.basename(config_filename))
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(config_header + yaml_config)
        return full_path
