"""SWE-bench processer — preprocess prompts, skip judge, compute patch rate."""

from ...utils import get_logger
from ..data import EvaluationSample
from .base_processor import BaseProcesser

logger = get_logger(__name__)

# Prompt template inspired by SWE-agent's default instance_template.
_SWE_BENCH_PROMPT = """\
You are working on the repository `{repo}` (checked out at `/testbed`).

The following issue has been reported:

{problem_statement}

Your task:
1. Explore the repository to understand the relevant code.
2. Make the minimal changes necessary to fix the issue.
3. Verify your fix by running the relevant tests.

Important:
- Do NOT modify test files unless the issue specifically requires it.
- Keep your changes focused and minimal.
"""


class SWEBenchProcesser(BaseProcesser):
    """Processer for SWE-bench evaluation."""

    name: str = "SWEBench"

    def preprocess_one(self, sample: EvaluationSample, recorder=None) -> EvaluationSample:
        """Build the augmented question with repo context."""
        _ = recorder
        meta = sample.meta or {}
        augmented = _SWE_BENCH_PROMPT.format(
            repo=meta.get("repo", "unknown"),
            problem_statement=sample.raw_question,
        )
        sample.update(augmented_question=augmented)
        return sample

    async def judge_one(self, sample: EvaluationSample) -> EvaluationSample:
        """Skip online judging — SWE-bench evaluation is done offline."""
        # Mark as not judged (None = unknown).
        sample.update(correct=None)
        return sample

    def calculate_metrics(self, samples: list[EvaluationSample]) -> dict:
        """Compute patch rate: fraction of samples that produced a non-empty patch."""
        total = len(samples)
        if total == 0:
            return {"total": 0, "patch_count": 0, "patch_rate": 0.0}

        patch_count = sum(1 for s in samples if s.extracted_final_answer and s.extracted_final_answer.strip())
        return {
            "total": total,
            "patch_count": patch_count,
            "patch_rate": round(patch_count / total, 4),
        }
