import json
from datetime import datetime

DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class StringUtils:
    @staticmethod
    def indent_lines(lines: str | list[str], indent: int = 2) -> str:
        """Add indentation to each line of text."""
        if isinstance(lines, str):
            lines = lines.split("\n")
        return "\n".join(" " * indent + line for line in lines)

    @staticmethod
    def remove_newlines(s: str) -> str:
        """Replace all newline characters with spaces."""
        return s.replace("\n", " ")

    @staticmethod
    def to_json_string(obj: dict | list, indent: int = None) -> str:
        """Convert object to JSON formatted string."""
        return json.dumps(obj, indent=indent, ensure_ascii=False)

    @staticmethod
    def truncate_str(text: str | dict, max_length: int = 100, oneline: bool = True) -> str:
        """Truncate text to max_length, adding ellipsis if truncated.

        Args:
            text: The text to truncate
            max_length: Maximum length of the returned text
            oneline: Whether to convert text to a single line by removing newlines
        """
        if isinstance(text, dict | list):
            text = StringUtils.to_json_string(text)
        if oneline:
            text = " ".join(text.splitlines())
        if len(text) <= max_length:
            return text
        else:
            return text[: max_length - 3] + "..."

    @staticmethod
    def timestamp_to_datetime(timestamp: int, is_ms: bool = False, format_str: str = DEFAULT_DATETIME_FORMAT) -> str:
        """Convert timestamp to formatted datetime string."""
        if is_ms or timestamp > 1e12:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp).strftime(format_str)

    @staticmethod
    def get_current_datetime_str(format_str: str = DEFAULT_DATETIME_FORMAT) -> str:
        return datetime.now().strftime(format_str)

    @staticmethod
    def get_current_timestamp(is_ms: bool = False) -> int:
        ts = datetime.now().timestamp()
        if is_ms:
            ts *= 1000
        return int(ts)
