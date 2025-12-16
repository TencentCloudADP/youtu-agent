import re

from ...config import EvalConfig
from ...utils import SimplifiedAsyncOpenAI, get_logger
from ..data import EvaluationSample
from .base_processor import BaseProcesser
from .prompts import AUGMENTATION_PROMPTS, JUDGE_PROMPTS
from .utils import MetricsUtils

logger = get_logger(__name__)


JUDGE_PROMPT_MAP = {
    "default": JUDGE_PROMPTS["default"],
    "BrowseComp": JUDGE_PROMPTS["default"],
    "GAIA": JUDGE_PROMPTS["default"],
    "BrowseComp_ZH": JUDGE_PROMPTS["BrowseComp_ZH"],
    "WebWalkerQA": JUDGE_PROMPTS["WebWalkerQA"],
    "XBench": JUDGE_PROMPTS["XBench"],
}


class BaseLLMJudgeProcesser(BaseProcesser):
    """Base class for processers that use LLM for judging."""

    name = "default"

    def __init__(self, config: EvalConfig) -> None:
        super().__init__(config)
        self.judge_client = SimplifiedAsyncOpenAI(**config.judge_model.model_provider.model_dump())

    def preprocess_one(self, sample: EvaluationSample) -> EvaluationSample:
        """Preprocess a single sample."""
        question = sample.raw_question
        template = AUGMENTATION_PROMPTS.get(self.name, AUGMENTATION_PROMPTS["default"])
        augmented_question = template.format(question=question)
        sample.update(
            augmented_question=augmented_question,
        )
        return sample

    async def judge_one(self, data: EvaluationSample) -> EvaluationSample:
        """Judge a single sample."""
        question = data.raw_question
        response = data.response
        correct_answer = data.correct_answer or "unknown"

        if correct_answer == "unknown":
            # if correct answer is unknown, we cannot judge
            data.update(judged_response="invalid", correct=False, reward=0.0)
            return data

        # if exact match, return directly(maybe extract exact answer from response first)
        if self._extract_exact_answer(response) == correct_answer:
            data.update(judged_response="Exact match", correct=True, reward=1.0)
            return data

        messages = self._get_judge_messages(question=question, response=response, correct_answer=correct_answer)
        content = await self.judge_client.query_one(
            messages=messages, **self.config.judge_model.model_params.model_dump()
        )
        parsed_content = self._parse_judge_response(content)

        data.judged_response = content
        # update the return data with parsed content
        data.update(**parsed_content)
        return data

    def calculate_metrics(self, samples: list[EvaluationSample]) -> dict:
        """Caculate metrics from the judged data."""
        return {
            **MetricsUtils.calculate_pass_at_k_metrics(samples, k=self.config.pass_k),
            **MetricsUtils.calculate_level_pass_at_k_metrics(samples, k=self.config.pass_k),
        }

    def _get_judge_messages(self, question: str, response: str, correct_answer: str) -> list:
        if self.name not in JUDGE_PROMPT_MAP:
            logger.warning(f"Judge prompt for {self.name} is not implemented! Using default judge prompt.")
        template = JUDGE_PROMPT_MAP.get(self.name, JUDGE_PROMPT_MAP["default"])
        input = template.format(question=question, response=response, correct_answer=correct_answer)
        return [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": input}]

    def _parse_judge_response(self, response: str) -> dict:
        """Parse the judge response into a structured format."""
        pattern = re.compile(
            r"(?=.*?extracted_final_answer:\s*(?P<extracted_final_answer>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?reasoning:\s*(?P<reasoning>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?correct:\s*(?P<correct>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?confidence:\s*(?P<confidence>\d+)\s*%?(?=\n\s*\w+:|$))?",
            re.DOTALL,
        )
        # remove the bold formatting
        response = response.replace("**", "")
        # search for the pattern in the response
        match = pattern.search(response)
        if not match:
            raise ValueError("Invalid judge response format.")

        return {
            "extracted_final_answer": match.group("extracted_final_answer").strip()
            if match.group("extracted_final_answer")
            else "",
            "reasoning": match.group("reasoning").strip() if match.group("reasoning") else "",
            "correct": match.group("correct").strip().lower() == "yes" if match.group("correct") else False,
            "confidence": int(match.group("confidence")) if match.group("confidence") else None,
        }

    def _extract_exact_answer(self, response: str) -> str:
        """Extract the exact answer from the response."""
        return response.strip() if response else ""
