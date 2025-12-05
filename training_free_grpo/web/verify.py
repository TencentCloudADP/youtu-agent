import re

from training_free_grpo.llm import LLM
from training_free_grpo.web.prompts import WEB_JUDGE_TEMPLATE

llm = LLM()


def verify_func(sample: dict, ground_truth: str, timeout_score: float = 0) -> float:
    """judge the response is correct or not based on LLM"""
    try:
        # get the response from LLM
        response = llm.chat(
            WEB_JUDGE_TEMPLATE.format(problem=sample["problem"], answer=ground_truth, response=sample["response"])
        )
        # parse the response
        pattern = re.compile(
            r"(?=.*?EXPLANATION:\s*(?P<reasoning>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?GRADE:\s*(?P<correct>.*?)(?=\n\s*\w+:|$))?",
            re.DOTALL,
        )
        response = response.replace("**", "")
        match = pattern.search(response)
        reasoning = match.group("reasoning").strip() if match.group("reasoning") else ""
        correct = match.group("correct").strip().upper() == "CORRECT" if match.group("correct") else False
        return float(correct)

    except Exception as e:
        print(f"Warning: failed in verifying response, {e}")
        return 0.0
