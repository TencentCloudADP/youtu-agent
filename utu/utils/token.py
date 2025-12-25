try:
    import tiktoken

    _tokenizer = tiktoken.get_encoding("cl100k_base")
except ImportError:
    pass


class TokenUtils:
    @staticmethod
    def truncate_text_by_token(text: str, limit: int = -1) -> str:
        """Truncate text to a given token limit with tiktoken."""
        if limit <= 0 or not text:
            return text
        tokens = _tokenizer.encode(text)
        if len(tokens) <= limit:
            return text
        truncated_tokens = tokens[:limit]
        truncated_text = _tokenizer.decode(truncated_tokens)
        return truncated_text + "..."

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text))
