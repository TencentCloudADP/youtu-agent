import re

from ..data import EvaluationSample
from .base_llm_processor import BaseLLMJudgeProcesser
from .utils import MetricsUtils


class WebWalkerQAProcesser(BaseLLMJudgeProcesser):
    """Processer for WebWalkerQA evaluation."""

    name: str = "WebWalkerQA"

    def preprocess_one(self, sample: EvaluationSample) -> EvaluationSample:
        # MODE 1: use root_url
        # aug_question = f"{sample.raw_question}\nReference url: {sample.meta['root_url']}"
        # MODE 2: w/o
        aug_question = sample.raw_question
        sample.update(augmented_question=aug_question)
        return sample

    def calculate_metrics(self, samples: list[EvaluationSample]) -> dict:
        """Calculate metrics from the judged data."""
        return {
            **MetricsUtils.calculate_pass_at_k_metrics(samples, k=self.config.pass_k),
            **MetricsUtils.calculate_level_pass_at_k_metrics(samples, k=self.config.pass_k),
        }

    def _parse_judge_response(self, response: str) -> dict:
        """Parse the judge response into a structured format."""
        pattern = re.compile(
            r"(?=.*?EXPLANATION:\s*(?P<reasoning>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?GRADE:\s*(?P<correct>.*?)(?=\n\s*\w+:|$))?",
            re.DOTALL,
        )
        # remove the bold formatting
        response = response.replace("**", "")
        match = pattern.search(response)
        if not match:
            raise ValueError("Invalid judge response format.")

        return {
            "reasoning": match.group("reasoning").strip() if match.group("reasoning") else "",
            "correct": match.group("correct").strip() == "CORRECT" if match.group("correct") else False,
        }
