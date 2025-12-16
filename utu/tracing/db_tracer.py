from typing import Any

from agents.tracing import Span, Trace, TracingProcessor, get_current_trace
from agents.tracing.span_data import (
    FunctionSpanData,
    GenerationSpanData,
    ResponseSpanData,
)

from ..db import GenerationTracingModel, ToolTracingModel
from ..utils import OpenAIUtils, SQLModelUtils, get_logger

logger = get_logger(__name__)


class DBTracingProcessor(TracingProcessor):
    """Basic tracing processor that stores events into database.

    Required environment variables: `UTU_DB_URL`
    """

    def __init__(self) -> None:
        if not SQLModelUtils.check_db_available():
            logger.warning("UTU_DB_URL not set or connection failed! Tracing will not be stored into database!")
            self.enabled = False
        else:
            self.enabled = True

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        if not self.enabled:
            return

        data = span.span_data
        if isinstance(data, GenerationSpanData):
            with SQLModelUtils.create_session() as session:
                session.add(
                    GenerationTracingModel(
                        trace_id=get_current_trace().trace_id,
                        span_id=span.span_id,
                        input=data.input,
                        output=data.output,
                        model=data.model,
                        model_configs=data.model_config,
                        usage=data.usage,
                    )
                )
                session.commit()
        elif isinstance(data, ResponseSpanData):
            # print(f"> response_id={data.response.id}: {data.response.model_dump()}")
            with SQLModelUtils.create_session() as session:
                session.add(
                    GenerationTracingModel(
                        trace_id=get_current_trace().trace_id,
                        span_id=span.span_id,
                        input=data.input,
                        output=OpenAIUtils.get_response_output(data.response),
                        model=OpenAIUtils.maybe_basemodel_to_dict(data.response.model),
                        model_configs=OpenAIUtils.get_response_configs(data.response),
                        usage=OpenAIUtils.maybe_basemodel_to_dict(data.response.usage),
                        type="responses",
                        response_id=data.response.id,
                    )
                )
                session.commit()
        elif isinstance(data, FunctionSpanData):
            with SQLModelUtils.create_session() as session:
                session.add(
                    ToolTracingModel(
                        name=data.name,
                        input=data.input,
                        output=data.output,
                        mcp_data=data.mcp_data,
                        trace_id=get_current_trace().trace_id,
                        span_id=span.span_id,
                    )
                )
                session.commit()

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass
