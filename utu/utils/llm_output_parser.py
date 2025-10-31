import json
import re

LANGUAGE_TO_CANDS = {
    "python": ["py", "python"],
    "json": ["json"],
    "javascript": ["js", "javascript"],
    "yaml": ["yaml", "yml"],
}


class LLMOutputParser:
    @staticmethod
    def extract_code_blocks(s: str) -> list[tuple[str, str]]:
        """Parse all code blocks (language, code) from the given string."""
        pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
        code_blocks = []
        for match in pattern.finditer(s):
            language = match.group(1).strip()
            code = match.group(2).strip()
            code_blocks.append((language, code))
        return code_blocks

    @staticmethod
    def extract_code_block_with_language(s: str, language: str) -> str:
        code_blocks = LLMOutputParser.extract_code_blocks(s)
        if not code_blocks:
            return s  # fallback
        for lang, code in code_blocks:
            if lang.lower() in LANGUAGE_TO_CANDS[language]:
                return code
        return code_blocks[0][1]  # fallback

    @staticmethod
    def extract_code_python(s: str) -> str:
        """Extract the first code block with the given language from the given string."""
        return LLMOutputParser.extract_code_block_with_language(s, "python")

    @staticmethod
    def extract_code_json(s: str, try_parse: bool = True) -> str | dict:
        """Extract the first json code block from the given string."""
        code = LLMOutputParser.extract_code_block_with_language(s, "json")
        if try_parse:
            try:
                return json.loads(code)
            except json.JSONDecodeError:
                return code
        return code

    @staticmethod
    def camel_to_snake(name: str) -> str:
        """Convert CamelCase / PascalCase to snake_case."""
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()

    @staticmethod
    def snake_to_camel(name: str, pascal: bool = True) -> str:
        """Convert snake_case to CamelCase / PascalCase."""
        components = name.split("_")
        if pascal:
            return "".join(x.title() for x in components)
        else:
            return components[0].lower() + "".join(x.title() for x in components[1:])

    @staticmethod
    def extract_float_number(s: str, return_last: bool = True) -> float | None:
        """Extract numbers from the given string.
        Args:
            return_last: if True, return the last number found; otherwise return first number found.
        """
        pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
        matches = pattern.findall(s)
        if not matches:
            return None
        return float(matches[-1]) if return_last else float(matches[0])
