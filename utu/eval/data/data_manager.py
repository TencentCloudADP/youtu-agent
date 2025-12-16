import abc
from typing import Literal

from sqlmodel import select

from ...config import EvalConfig
from ...db import DatasetSample, EvaluationSample
from ...utils import SQLModelUtils, get_logger

logger = get_logger(__name__)


class BaseDataManager(abc.ABC):
    """Base data manager for loading and saving data."""

    data: list[EvaluationSample]

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def load(self) -> list[EvaluationSample]:
        """Load the dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, **kwargs) -> None:
        """Save the dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_samples(self, stage: Literal["init", "rollout", "judged"] = None) -> list[EvaluationSample]:
        """Get samples of specified stage from the dataset."""
        raise NotImplementedError


class DBDataManager(BaseDataManager):
    """Database data manager for loading and saving data."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def load(self) -> list[EvaluationSample]:
        if self._check_exp_id():
            logger.warning(f"exp_id {self.config.exp_id} already exists in db")
            return self.get_samples()

        with SQLModelUtils.create_session() as session:
            datapoints = session.exec(
                select(DatasetSample).where(DatasetSample.dataset == self.config.data.dataset)
            ).all()
            logger.info(f"Loaded {len(datapoints)} samples from {self.config.data.dataset}.")
            samples = []
            logger.info(f"Duplicate {self.config.pass_k} times for each sample.")
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
                        exp_id=self.config.exp_id,  # add exp_id
                    )
                    samples.append(sample)
            logger.info(f"Created {len(samples)} samples for exp_id {self.config.exp_id}.")

            self.data = samples
            self.save(self.data)  # save to db
            return self.data

    def get_samples(
        self, stage: Literal["init", "rollout", "judged"] = None, limit: int = None
    ) -> list[EvaluationSample]:
        """Get samples from exp_id with specified stage."""
        with SQLModelUtils.create_session() as session:
            samples = session.exec(
                select(EvaluationSample)
                .where(
                    EvaluationSample.exp_id == self.config.exp_id,
                    EvaluationSample.stage == stage if stage else True,
                )
                .order_by(EvaluationSample.dataset_index)
                .limit(limit)
            ).all()
            return samples

    def save(self, samples: list[EvaluationSample] | EvaluationSample) -> None:
        """Update or add sample(s) to db."""
        if isinstance(samples, list):
            with SQLModelUtils.create_session() as session:
                session.add_all(samples)
                session.commit()
        else:
            with SQLModelUtils.create_session() as session:
                session.add(samples)
                session.commit()

    def delete_samples(self, samples: list[EvaluationSample] | EvaluationSample) -> None:
        """Delete sample(s) from db."""
        if isinstance(samples, list):
            with SQLModelUtils.create_session() as session:
                for sample in samples:
                    session.delete(sample)
                session.commit()
        else:
            with SQLModelUtils.create_session() as session:
                session.delete(samples)
                session.commit()

    def _check_exp_id(self) -> bool:
        # check if any record has the same exp_id
        with SQLModelUtils.create_session() as session:
            has_exp_id = session.exec(
                select(EvaluationSample).where(EvaluationSample.exp_id == self.config.exp_id)
            ).first()
        return has_exp_id is not None
