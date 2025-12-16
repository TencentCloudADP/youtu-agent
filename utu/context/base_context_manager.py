from agents import RunContextWrapper, TContext, TResponseInputItem

from ..utils import get_logger

logger = get_logger(__name__)


class BaseContextManager:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def preprocess(
        self, input: str | list[TResponseInputItem], run_context: RunContextWrapper[TContext] = None
    ) -> str | list[TResponseInputItem]:
        return input


class DummyContextManager(BaseContextManager):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.max_turn_stop_msg = (
            "You have reached the maximum number of turns allowed. "
            "Please DO NOT use ANY tools, provide your final answer."
        )

    def preprocess(
        self, input: str | list[TResponseInputItem], run_context: RunContextWrapper[TContext] = None
    ) -> str | list[TResponseInputItem]:
        # NOTE: filter type="reasoning" items for vllm cannot process it for now!
        # return ChatCompletionConverter.filter_items(input)

        # handle MaxTurnsExceeded
        context: dict = run_context.context
        assert isinstance(context, dict), "run_context.context should be a dict"
        current_turn, max_turns = context.get("current_turn"), context.get("max_turns")
        logger.debug(f"Current turn: {current_turn}, Max turns: {max_turns}")
        if current_turn is not None and max_turns is not None and current_turn == max_turns:
            logger.warning(f"Max turns {max_turns} encountered! Injecting stop message.")
            input.append(
                {
                    "role": "user",
                    "content": self.max_turn_stop_msg,
                }
            )
        return input
