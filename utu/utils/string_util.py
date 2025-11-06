import json


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
    def to_json_string(obj: dict, indent: int = None) -> str:
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
        if isinstance(text, dict):
            text = StringUtils.to_json_string(text)
        if oneline:
            text = " ".join(text.splitlines())
        if len(text) <= max_length:
            return text
        else:
            return text[: max_length - 3] + "..."
