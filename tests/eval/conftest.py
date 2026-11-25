import os

# utu package import asserts these; dummy values keep eval unit tests offline.
os.environ.setdefault("UTU_LLM_TYPE", "chat.completions")
os.environ.setdefault("UTU_LLM_MODEL", "dummy")
os.environ.setdefault("UTU_LLM_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("UTU_LLM_API_KEY", "dummy")
