import abc

from ...config import EvalConfig
from ..data import EvaluationSample


class BaseProcesser:
    """Base class for processers in evaluation tasks.

    Each processer implements the following evaluation phases:
      - load: load and process data (if necessary)
      - judge: judge the correctness of a batch of predictions
      - stat: get metrics.
    """

    name: str = None
    config: EvalConfig = None

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def preprocess_one(self, sample: EvaluationSample, recorder=None) -> EvaluationSample:
        """Preprocess a single sample.

        Args:
            sample: EvaluationSample to preprocess
            recorder: Optional practice TaskRecorder. Ignored by evaluation
                processors; TrainingFreeGRPOProcesser uses it to attach experiences.
                RolloutManager always passes this argument.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def judge_one(self, sample: EvaluationSample) -> EvaluationSample:
        """Judge a single sample."""
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_metrics(self, samples: list[EvaluationSample]) -> dict:
        """Calculate metrics from the judged data."""
        raise NotImplementedError

    async def stat(self, samples: list[EvaluationSample]) -> dict:
        metrics = self.calculate_metrics(samples)
        return {"benchmark": self.name, "metrics": metrics}
