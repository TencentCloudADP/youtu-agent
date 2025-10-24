import json
import logging
import pathlib
from logging.handlers import TimedRotatingFileHandler
from typing import Literal

from colorlog import ColoredFormatter

DIR_LOGS = pathlib.Path(__file__).parent.parent.parent / "logs"
DIR_LOGS.mkdir(exist_ok=True)

# Flag to track if logging has been set up
_LOGGING_INITIALIZED = False


def setup_logging(level: Literal["WARNING", "INFO", "DEBUG"] = "WARNING") -> None:
    # Check if logging has already been initialized
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        logging.getLogger().warning("Logging has already been initialized! Skipping...")
        return

    utu_logger = logging.getLogger("utu")
    utu_logger.setLevel(logging.DEBUG)
    if utu_logger.handlers:
        utu_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    color_formatter = ColoredFormatter(
        "%(green)s%(asctime)s%(reset)s[%(blue)s%(name)s%(reset)s] - "
        "%(log_color)s%(levelname)s%(reset)s - %(filename)s:%(lineno)d - %(green)s%(message)s%(reset)s",
        # " - %(cyan)s%(threadName)s%(reset)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={"asctime": {"green": "green"}, "name": {"blue": "blue"}},
    )
    console_handler.setFormatter(color_formatter)

    file_handler = TimedRotatingFileHandler(
        DIR_LOGS / "utu.log", when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s[%(name)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s - %(threadName)s"
    )
    file_handler.setFormatter(formatter)

    utu_logger.addHandler(console_handler)
    utu_logger.addHandler(file_handler)
    utu_logger.info(f"Logging initialized with level {level}.")

    # Mark logging as initialized
    _LOGGING_INITIALIZED = True


def get_logger(name: str, level: int | str = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    def log_error_with_exc(msg, *args, **kwargs):
        kwargs["exc_info"] = True
        logger.error(msg, *args, **kwargs)

    logger.error_exc = log_error_with_exc
    return logger


def oneline_object(obj: object, limit: int = 100) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        s = json.dumps(str(obj), ensure_ascii=False)
    return f"{s[:limit]}..." if len(s) > limit else s
