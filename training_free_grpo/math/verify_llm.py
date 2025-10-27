import re
from training_free_grpo.llm import LLM


JUDGE_PROMPT = """You are a strict math evaluator.

Task: Compare the model's final answer to the golden answer and judge CORRECT or WRONG. Then provide a brief remark explaining the decision.

Return in this format only:
EXPLANATION: <one or two sentences>
GRADE: <CORRECT|WRONG>

Problem:
{problem}

Golden Answer:
{golden}

Model Output:
{output}
"""


_llm = LLM()


def verify_func(sample: dict, ground_truth: str | None = None, timeout_score: float = 0) -> float:
    """Model-based verifier for math answers using an LLM.

    - Prefers sample["golden_answer"] if present; otherwise falls back to provided ground_truth.
    - Returns 1.0 for CORRECT, 0.0 for WRONG; attaches remark in sample["judge_remark"].
    """
    golden = sample.get("golden_answer") or ground_truth or ""
    problem = sample.get("problem", "")
    output = sample.get("response", "")

    try:
        prompt = JUDGE_PROMPT.format(problem=problem, golden=golden, output=output)
        res = _llm.chat(prompt, temperature=0)
        pattern = re.compile(
            r"(?=.*?EXPLANATION:\s*(?P<reasoning>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?GRADE:\s*(?P<correct>.*?)(?=\n\s*\w+:|$))?",
            re.DOTALL,
        )
        match = pattern.search(res or "")
        reasoning = match.group("reasoning").strip() if match and match.group("reasoning") else ""
        correct = (match.group("correct").strip().upper() == "CORRECT") if match and match.group("correct") else False
        sample["judge_remark"] = reasoning
        return float(bool(correct))
    except Exception as e:
        sample["judge_remark"] = f"judge_error: {e}"
        return 0.0


