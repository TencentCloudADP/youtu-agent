import re

from utu.db import EvaluationSample
from utu.utils import FileUtils, SimplifiedAsyncOpenAI


async def verify_func(sample: EvaluationSample, timeout_score: float = 0, **kwargs) -> dict:
    """judge the response is correct or not based on LLM"""
    try:
        llm = kwargs.get("llm", SimplifiedAsyncOpenAI())
        template = FileUtils.load_prompts("practice/verify.yaml")["WEBWALKER_JUDGE_TEMPLATE"]
        # get the response from LLM
        up = FileUtils.get_jinja_template_str(template).render(
            problem=sample.raw_question, answer=sample.correct_answer, response=sample.response
        )
        response = await llm.query_one(messages=up)
        # parse the response
        pattern = re.compile(
            r"(?=.*?EXPLANATION:\s*(?P<reasoning>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?GRADE:\s*(?P<correct>.*?)(?=\n\s*\w+:|$))?",
            re.DOTALL,
        )
        response = response.replace("**", "")
        match = pattern.search(response)
        # reasoning = match.group("reasoning").strip() if match.group("reasoning") else ""
        correct = match.group("correct").strip().upper() == "CORRECT" if match.group("correct") else False
        return {
            "reward": float(correct),
            "reasoning": None,  # Not needed in web searching tasks
        }

    except Exception as e:
        print(f"Warning: failed in verifying response, {e}")
        return {"reward": timeout_score, "reasoning": None}
