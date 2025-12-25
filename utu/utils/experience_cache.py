import time
from datetime import datetime
from typing import Any

from sqlmodel import select

from ..db import ExperienceCacheModel
from .log import get_logger
from .sqlmodel_utils import SQLModelUtils

logger = get_logger(__name__)


class ExperienceCache:
    """Database-based cache for training experiences."""

    @staticmethod
    def save_experiences(
        experiment_name: str,
        step: int,
        experiences: dict[str, Any],
        epoch: int | None = None,
        batch: int | None = None,
    ) -> bool:
        """Save experiences to database.

        Args:
            experiment_name: Name of the experiment
            step: Step number
            experiences: Experience data to cache
            epoch: Epoch number (optional)
            batch: Batch number (optional)
            execution_time: Execution time in seconds (optional)

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            with SQLModelUtils.create_session() as session:
                # Check if record already exists
                stmt = select(ExperienceCacheModel).where(
                    ExperienceCacheModel.experiment_name == experiment_name, ExperienceCacheModel.step == step
                )
                existing_record = session.exec(stmt).first()

                if existing_record:
                    logger.debug(f"Experience cache for {experiment_name} step {step} already exists, updating.")
                    existing_record.experiences = experiences
                    existing_record.timestamp = time.time()
                    existing_record.datetime = datetime.now().isoformat()
                    if epoch is not None:
                        existing_record.epoch = epoch
                    if batch is not None:
                        existing_record.batch = batch
                else:
                    # Create new record
                    cache_record = ExperienceCacheModel(
                        experiment_name=experiment_name,
                        step=step,
                        epoch=epoch,
                        batch=batch,
                        experiences=experiences,
                        timestamp=time.time(),
                        datetime=datetime.now().isoformat(),
                    )
                    session.add(cache_record)

                session.commit()
                logger.debug(f"Cached experiences for {experiment_name} step {step} to database")
                return True

        except Exception as e:
            logger.error(f"Failed to save experiences to database: {e}")
            return False

    @staticmethod
    def load_experiences(experiment_name: str, step: int) -> dict[str, Any] | None:
        """Load experiences from database.

        Args:
            experiment_name: Name of the experiment
            step: Step number

        Returns:
            Dict[str, Any]: Cached experience data, None if not found
        """
        try:
            with SQLModelUtils.create_session() as session:
                stmt = select(ExperienceCacheModel).where(
                    ExperienceCacheModel.experiment_name == experiment_name, ExperienceCacheModel.step == step
                )
                record = session.exec(stmt).first()

                if record:
                    logger.debug(f"🔄 Using cached experiences for {experiment_name} step {step} from database")
                    return record.experiences
                else:
                    logger.debug(f"No cached experiences found for {experiment_name} step {step}")
                    return None

        except Exception as e:
            logger.error(f"Failed to load experiences from database: {e}")
            return None

    @staticmethod
    def exists(experiment_name: str, step: int) -> bool:
        """Check if experiences exist in cache.

        Args:
            experiment_name: Name of the experiment
            step: Step number

        Returns:
            bool: True if exists, False otherwise
        """
        try:
            with SQLModelUtils.create_session() as session:
                stmt = select(ExperienceCacheModel).where(
                    ExperienceCacheModel.experiment_name == experiment_name, ExperienceCacheModel.step == step
                )
                record = session.exec(stmt).first()
                return record is not None

        except Exception as e:
            logger.error(f"Failed to check experience cache existence: {e}")
            return False

    @staticmethod
    def delete_experiment_cache(experiment_name: str) -> bool:
        """Delete all cached experiences for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            with SQLModelUtils.create_session() as session:
                stmt = select(ExperienceCacheModel).where(ExperienceCacheModel.experiment_name == experiment_name)
                records = session.exec(stmt).all()

                for record in records:
                    session.delete(record)

                session.commit()
                logger.info(f"Deleted {len(records)} experience cache records for {experiment_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete experiment cache: {e}")
            return False
