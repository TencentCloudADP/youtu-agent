from collections.abc import Callable
from functools import wraps
from typing import Any

from sqlmodel import SQLModel, select

from ..utils import SQLModelUtils, get_logger

logger = get_logger(__name__)


def require_db(safe: bool = True) -> Callable:
    """Decorator to check database availability before executing database operations.

    Args:
        safe: If True, return None on failure; if False, raise exception
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not SQLModelUtils.check_db_available():
                if safe:
                    logger.debug(f"Database not available, skipping {func.__name__}")
                    return None if func.__name__ != "add" else False
                else:
                    raise RuntimeError("Database is not available")

            try:
                return func(*args, **kwargs)
            except Exception as e:
                if safe:
                    logger.warning(f"Database operation {func.__name__} failed: {e}")
                    return None if func.__name__ != "add" else False
                else:
                    raise

        return wrapper

    return decorator


class DBService:
    @staticmethod
    @require_db(safe=True)
    def add(data: SQLModel | list[SQLModel]) -> bool:
        """Add one or more records to database.

        Args:
            data: SQLModel instance(s) to add

        Returns:
            bool: True if successfully saved, False if failed
        """
        # Normalize input to list
        if isinstance(data, SQLModel):
            data = [data]
        elif isinstance(data, list):
            if not data:
                logger.warning("Empty data list provided")
                return False
            assert isinstance(data[0], SQLModel), "data must be a SQLModel or list of SQLModel"
        else:
            raise ValueError("data must be a SQLModel or list of SQLModel")

        # Save to database
        with SQLModelUtils.create_session() as session:
            session.add_all(data)
            session.commit()
        logger.debug(f"Successfully saved {len(data)} record(s) to database")
        return True

    @staticmethod
    @require_db(safe=True)
    def query(model_class: type[SQLModel], filters: dict = None) -> list[SQLModel]:
        """Query records with optional filters.

        Returns:
            list[SQLModel]: List of matching records, or None if DB unavailable
        """
        with SQLModelUtils.create_session() as session:
            stmt = select(model_class)
            if filters:
                for key, value in filters.items():
                    stmt = stmt.where(getattr(model_class, key) == value)
            return session.exec(stmt).all()

    @staticmethod
    @require_db(safe=True)
    def get_by_id(model_class: type[SQLModel], id: int) -> SQLModel | None:
        """Get a single record by ID.

        Returns:
            SQLModel | None: The record if found, None if not found or DB unavailable
        """
        with SQLModelUtils.create_session() as session:
            return session.get(model_class, id)

    @staticmethod
    @require_db(safe=True)
    def update(data: SQLModel) -> bool:
        """Update a record.

        Returns:
            bool: True if successfully updated, False if failed
        """
        with SQLModelUtils.create_session() as session:
            session.add(data)
            session.commit()
            session.refresh(data)
        return True

    @staticmethod
    @require_db(safe=True)
    def delete(data: SQLModel) -> bool:
        """Delete a record.

        Returns:
            bool: True if successfully deleted, False if failed
        """
        with SQLModelUtils.create_session() as session:
            session.delete(data)
            session.commit()
        return True
