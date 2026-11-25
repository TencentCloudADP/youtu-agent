"""RolloutManager always calls processer.preprocess_one(sample, recorder).

Standard evaluation processors historically only accepted `sample`, which
raises TypeError for every non-GRPO source (see #231).
"""

from utu.db.eval_datapoint import EvaluationSample
from utu.eval.processer.base_llm_processor import BaseLLMJudgeProcesser
from utu.eval.processer.gaia import GAIAProcesser
from utu.eval.processer.swe_bench import SWEBenchProcesser
from utu.eval.processer.web_walker import WebWalkerQAProcesser


class _DummyRecorder:
    experiences = {"e1": "remember to cite sources"}


def _sample(**kwargs) -> EvaluationSample:
    values = {"raw_question": "What is 2+2?", "file_name": "", "meta": {}}
    values.update(kwargs)
    return EvaluationSample(**values)


def _new(cls):
    # preprocess_one on these classes does not use judge_client / config,
    # so skip the heavy EvalConfig constructor.
    return object.__new__(cls)


def test_standard_processors_accept_positional_recorder():
    recorder = _DummyRecorder()
    cases = [
        (_new(BaseLLMJudgeProcesser), _sample(), "What is 2+2?"),
        (_new(GAIAProcesser), _sample(), "What is 2+2?"),
        (_new(WebWalkerQAProcesser), _sample(), "What is 2+2?"),
        (
            _new(SWEBenchProcesser),
            _sample(raw_question="the bug", meta={"repo": "org/repo"}),
            "org/repo",
        ),
    ]
    for processor, sample, expected_fragment in cases:
        # This is the exact calling convention in RolloutManager.preprocess_one.
        processed = processor.preprocess_one(sample, recorder)
        assert processed is sample
        assert expected_fragment in (sample.augmented_question or "")
