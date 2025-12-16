import random
from typing import Literal

from sqlmodel import select

from ..config import EvalConfig
from ..db import DatasetSample, EvaluationSample
from ..eval import DBDataManager
from ..utils import SQLModelUtils, get_logger

logger = get_logger(__name__)
random.seed(42)


class TrainingFreeGRPODataManager(DBDataManager):
    """Data manager for training-free GRPO data.

    This class extends DBDataManager to handle training-free GRPO-specific data loading. Inludes
    the ability to duplicate samples based on pass_k configuration, shuffle epoch data,
    load batch data, etc.
    """

    def __init__(self, config: EvalConfig) -> None:
        super().__init__(config)

    def load_epoch_data(self, epoch: int, shuffle: bool = True, truncate: int = None) -> list:
        """Load data for a specific epoch."""
        epoch_exp_id = f"{self.config.exp_id}_epoch_{epoch}"
        # check if epoch data exists
        if self._check_exp_id(epoch_exp_id):
            logger.warning(f"exp_id {epoch_exp_id} already exists in db")
            return self.get_batch_samples(epoch)

        with SQLModelUtils.create_session() as session:
            # Load all datapoints from the dataset
            datapoints = session.exec(
                select(DatasetSample).where(DatasetSample.dataset == self.config.data.dataset)
            ).all()
            logger.info(f"Loaded {len(datapoints)} samples from {self.config.data.dataset}.")

            # truncate the dataset if needed
            if truncate:
                datapoints = datapoints[:truncate]
                logger.info(f"Truncated dataset to first {truncate} samples.")

            # shuffle the original datapoints if needed
            if shuffle:
                random.shuffle(datapoints)
                logger.info("Shuffled the original datapoints.")

            samples = []
            logger.info(f"Duplicate {self.config.pass_k} times for each sample.")
            # Create duplicates for each datapoint, keeping duplicates adjacent
            for dp in datapoints:
                for _ in range(self.config.pass_k):
                    sample = EvaluationSample(
                        dataset=dp.dataset,
                        dataset_index=dp.index,
                        source=dp.source,
                        raw_question=dp.question,
                        level=dp.level,
                        correct_answer=dp.answer,
                        file_name=dp.file_name,
                        meta=dp.meta,
                        exp_id=epoch_exp_id,  # add exp_id
                    )
                    samples.append(sample)
            logger.info(f"Created {len(samples)} samples for exp_id {epoch_exp_id} with duplicates kept adjacent.")
            self.data = samples
            self.save(self.data)  # save to db
            return self.data

    def get_batch_samples(
        self,
        epoch: int,
        stage: Literal["init", "rollout", "judged"] = None,
        limit: int = None,
        batch_size: int = 64,
        batch_idx: int | None = None,
    ) -> list[EvaluationSample]:
        """Get samples for a specific batch."""
        exp_id = f"{self.config.exp_id}_epoch_{epoch}"
        with SQLModelUtils.create_session() as session:
            samples = session.exec(
                select(EvaluationSample)
                .where(
                    EvaluationSample.exp_id == exp_id,
                )
                .order_by(EvaluationSample.dataset_index)
                .limit(limit)
            ).all()
        if batch_idx is not None:
            batch_size = self.config.pass_k * batch_size
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            samples = samples[start_idx:end_idx]
        # select by stage
        if stage:
            samples = [s for s in samples if s.stage == stage]
        return samples

    def _check_exp_id(self, exp_id: str) -> bool:
        # check if any record has the same exp_id
        with SQLModelUtils.create_session() as session:
            has_exp_id = session.exec(select(EvaluationSample).where(EvaluationSample.exp_id == exp_id)).first()
        return has_exp_id is not None

    def check_dataset(self, dataset: str) -> bool:
        """Check if any record exists for the given dataset."""
        with SQLModelUtils.create_session() as session:
            has_exp_id = session.exec(select(DatasetSample).where(DatasetSample.dataset == dataset)).first()
        return has_exp_id is not None
