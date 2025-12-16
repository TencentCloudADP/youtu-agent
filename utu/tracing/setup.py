"""
open-inference: https://github.com/Arize-ai/openinference
https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-openai-agents
session-level tracing in @phoenix https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-sessions
"""

from agents import add_trace_processor, set_tracing_disabled
from openinference.instrumentation.openai import OpenAIInstrumentor

# from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
# from phoenix.otel import TracerProvider, register
from openinference.semconv.resource import ResourceAttributes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import Resource, TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from ..utils import EnvUtils, SQLModelUtils, get_logger
from .db_tracer import DBTracingProcessor
from .otel_agents_instrumentor import OpenAIAgentsInstrumentor

logger = get_logger(__name__)

OTEL_TRACING_PROVIDER: TracerProvider | None = None
DB_TRACING_PROCESSOR: DBTracingProcessor | None = None


def setup_otel_tracing(
    endpoint: str = None,
    project_name: str = None,
    debug: bool = False,
) -> None:
    """Setup OpenTelemetry tracing. We use arize-phoenix by default, see
    https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-using-phoenix-otel for details.
    """
    global OTEL_TRACING_PROVIDER
    if OTEL_TRACING_PROVIDER is not None:
        logger.warning("OpenTelemetry tracing is already set up! Skipping...")
        return

    endpoint = endpoint or EnvUtils.get_env("PHOENIX_ENDPOINT", "")
    project_name = project_name or EnvUtils.get_env("PHOENIX_PROJECT_NAME", "")
    if not endpoint or not project_name:
        logger.warning("PHOENIX_ENDPOINT or PHOENIX_PROJECT_NAME is not set! Skipping OpenTelemetry tracing.")
        set_tracing_disabled(True)  # we disable the openai's default tracing
        return

    # https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/custom-spans
    # create your key: https://app.phoenix.arize.com/s/_space_name_/settings/general
    if endpoint.startswith("https://app.phoenix.arize.com"):
        api_key = EnvUtils.get_env("PHOENIX_API_KEY", "")
        if not api_key:
            logger.warning(f"You use PHOENIX_ENDPOINT={endpoint} but PHOENIX_API_KEY is not set! Tracing may not work.")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    else:
        headers = None
    logger.info(f"Setting up OpenTelemetry tracing with endpoint: {endpoint}, project name: {project_name}")
    OTEL_TRACING_PROVIDER = TracerProvider(resource=Resource({ResourceAttributes.PROJECT_NAME: project_name}))
    OTEL_TRACING_PROVIDER.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers)))
    if debug:
        OTEL_TRACING_PROVIDER.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    # instrument
    OpenAIInstrumentor().instrument(tracer_provider=OTEL_TRACING_PROVIDER)
    # use `set_trace_processors` instead of `add_trace_processor` to remove default processors
    OpenAIAgentsInstrumentor().instrument(tracer_provider=OTEL_TRACING_PROVIDER, exclusive_processor=True)


def setup_db_tracing() -> None:
    """Setup DB tracing."""
    global DB_TRACING_PROCESSOR
    if DB_TRACING_PROCESSOR is not None:
        logger.warning("DB tracing is already set up! Skipping...")
        return

    if not SQLModelUtils.check_db_available():
        logger.warning("UTU_DB_URL not set or database connection failed! Tracing will not be stored into database!")
        return
    logger.info("Setting up DB tracing")
    DB_TRACING_PROCESSOR = DBTracingProcessor()
    add_trace_processor(DB_TRACING_PROCESSOR)  # add an additional processor


def setup_tracing() -> None:
    setup_otel_tracing()
    setup_db_tracing()
