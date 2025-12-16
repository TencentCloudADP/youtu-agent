from sqlmodel import Session, create_engine, text

from .env import EnvUtils
from .log import get_logger

logger = get_logger(__name__)


class SQLModelUtils:
    _engine = None  # singleton
    _db_available = None  # cache for database availability check
    _last_check_time = None  # timestamp of last check

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

    @classmethod
    def check_db_available(cls, force_check: bool = False, cache_ttl: int = 60) -> bool:
        """Check if database is available with caching.

        Args:
            force_check: Force a fresh check, bypassing cache
            cache_ttl: Cache time-to-live in seconds (default: 60s)

        Returns:
            bool: True if database is available, False otherwise
        """
        import time

        # Return cached result if available and not expired
        if not force_check and cls._db_available is not None and cls._last_check_time is not None:
            if time.time() - cls._last_check_time < cache_ttl:
                logger.debug(f"Using cached DB availability status: {cls._db_available}")
                return cls._db_available

        # Perform actual check
        logger.debug("Performing fresh database availability check")

        if not EnvUtils.get_env("UTU_DB_URL", ""):
            cls._db_available = False
            cls._last_check_time = time.time()
            return False

        try:
            engine = cls.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            cls._db_available = True
            cls._last_check_time = time.time()
            logger.debug("Database is available")
            return True
        except Exception as e:
            cls._db_available = False
            cls._last_check_time = time.time()
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
            from utu.db import eval_datapoint, tool_cache_model, tracing_model, trajectory_model  # noqa: F401
        except Exception as e:
            logger.debug(f"Model import warning (non-fatal): {e}")

        # Create tables if not exist
        try:
            from sqlmodel import SQLModel

            SQLModel.metadata.create_all(engine)
            logger.info("Database schema ensured (tables created if missing).")
        except Exception as e:
            logger.warning(f"SQLModel metadata create_all failed: {e}")
