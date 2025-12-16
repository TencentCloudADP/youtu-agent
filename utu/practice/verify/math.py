from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from utu.db import EvaluationSample


def verify_func(sample: EvaluationSample, timeout_score: float = 0, **kwargs) -> dict:
    model_output = sample.response
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth = sample.correct_answer
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return {"reward": float(ret_score), "reasoning": None}
