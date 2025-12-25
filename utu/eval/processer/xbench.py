import re

from ..data import EvaluationSample as Datapoint
from .base_llm_processor import BaseLLMJudgeProcesser
from .utils import MetricsUtils


class XBenchProcesser(BaseLLMJudgeProcesser):
    """Processer for XBench evaluation."""

    name: str = "XBench"

    def calculate_metrics(self, samples: list[Datapoint]) -> dict:
        """Calculate metrics for XBench evaluation."""
        # Calculate confidence metrics
        total = len(samples)
        for item in samples:
            if item.confidence is None:
                item.confidence = 100 if item.judged_response == "Exact match" else 0
        confidence_scores = [item.confidence for item in samples if item.judged_response != "invalid"]

        return {
            **MetricsUtils.calculate_pass_at_k_metrics(samples, k=self.config.pass_k),
            "Average Confidence (%)": round(sum(confidence_scores) / total, 2),
            **MetricsUtils.calculate_level_pass_at_k_metrics(samples, k=self.config.pass_k),
        }

    def _parse_judge_response(self, response: str) -> dict:
        """Parse the judge response into a structured format."""
        pattern = re.compile(
            r"(?=.*?最终答案:\s*(?P<extracted_final_answer>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?解释:\s*(?P<reasoning>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?结论:\s*(?P<correct>.*?)(?=\n\s*\w+:|$))?",
            re.DOTALL,
        )
        # remove the bold formatting
        response = response.replace("**", "")
        match = pattern.search(response)
        if not match:
            raise ValueError("Invalid judge response format.")

        return {
            "extracted_final_answer": match.group("extracted_final_answer").strip()
            if match.group("extracted_final_answer")
            else "",
            "reasoning": match.group("reasoning").strip() if match.group("reasoning") else "",
            "correct": match.group("correct").strip() == "正确" if match.group("correct") else False,
        }

    def _extract_exact_answer(self, response: str) -> str:
        """Earse the exact answer from the response."""
        pattern = re.compile(r"最终答案:\s*(.*)")
        match = pattern.search(response)
        if not match or not match.group(1):
            return ""
        return match.group(1).strip()
