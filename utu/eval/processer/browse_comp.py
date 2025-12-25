import re

from ..data import EvaluationSample as Datapoint
from .base_llm_processor import BaseLLMJudgeProcesser
from .utils import MetricsUtils


class BrowseCompProcesser(BaseLLMJudgeProcesser):
    """Processer for BrowseComp evaluation."""

    name: str = "BrowseComp"

    def calculate_metrics(self, samples: list[Datapoint]) -> dict:
        """Calculate metrics for BrowseComp evaluation."""
        return {
            **MetricsUtils.calculate_overall_metrics(samples),
            **MetricsUtils.calculate_level_metrics(samples),
        }

    def _extract_exact_answer(self, response: str) -> str:
        """Extract the exact answer from the response."""
        pattern = re.compile(r"Exact Answer:\s*(.*)")
        match = pattern.search(response)
        if not match or not match.group(1):
            return ""
        return match.group(1).strip()


class BrowseCompZHProcesser(BrowseCompProcesser):
    """Processer for BrowseCompZH evaluation."""

    name: str = "BrowseComp_ZH"

    def calculate_metrics(self, samples: list[Datapoint]) -> dict:
        """Calculate metrics from the judged data."""
        return {
            **MetricsUtils.calculate_pass_at_k_metrics(samples, k=self.config.pass_k),
            **MetricsUtils.calculate_calibration(samples),
            **MetricsUtils.calculate_level_pass_at_k_metrics(samples, k=self.config.pass_k),
        }
