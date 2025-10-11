from sqlmodel import Session, create_engine, text

from .env import EnvUtils
from .log import get_logger

logger = get_logger(__name__)


class SQLModelUtils:
    _engine = None  # singleton

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            cls._engine = create_engine(
                EnvUtils.get_env("UTU_DB_URL"),  # assert UTU_DB_URL is provided here!
                pool_size=300,
                max_overflow=500,
                pool_timeout=30,
                pool_pre_ping=True,
            )
            # Ensure DB schema/tables exist on first engine init
            try:
                cls._init_db_schema(cls._engine)
            except Exception as e:
                logger.warning(f"Auto schema creation skipped due to error: {e}")
        return cls._engine

    @staticmethod
    def create_session():
        return Session(SQLModelUtils.get_engine())

    @staticmethod
    def check_db_available():
        if not EnvUtils.get_env("UTU_DB_URL", ""):
            # logger.error("UTU_DB_URL is not set")
            return False
        try:
            engine = SQLModelUtils.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    @staticmethod
    def _init_db_schema(engine):
        """
        Import all SQLModel table definitions and create tables if they do not exist.
        """
        # Import models so SQLModel knows about all tables
        try:
            # Core models registered here
            from utu.db import eval_datapoint, tool_cache_model, tracing_model  # noqa: F401
        except Exception as e:
            logger.debug(f"Model import warning (non-fatal): {e}")

        # Create tables if not exist
        try:
            from sqlmodel import SQLModel

            SQLModel.metadata.create_all(engine)
            logger.info("Database schema ensured (tables created if missing).")
        except Exception as e:
            logger.warning(f"SQLModel metadata create_all failed: {e}")
