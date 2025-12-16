from collections.abc import Collection
from typing import Any, cast

from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import Tracer

from ..utils import get_logger

# from openinference.instrumentation.openai_agents.package import _instruments
# from openinference.instrumentation.openai_agents.version import __version__
_instruments = ("openai-agents >= 0.2.6",)
__version__ = "1.3.0"

logger = get_logger(__name__)


class OpenAIAgentsInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for openai-agents
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (exclusive_processor := kwargs.get("exclusive_processor")):
            exclusive_processor = False
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        # from openinference.instrumentation.openai_agents._processor import (
        #     OpenInferenceTracingProcessor,
        # )
        from .otel_agents_processor import OpenInferenceTracingProcessor

        if exclusive_processor:
            from agents import set_trace_processors

            set_trace_processors([OpenInferenceTracingProcessor(cast(Tracer, tracer))])
        else:
            from agents import add_trace_processor

            add_trace_processor(OpenInferenceTracingProcessor(cast(Tracer, tracer)))

    def _uninstrument(self, **kwargs: Any) -> None:
        # TODO
        pass
