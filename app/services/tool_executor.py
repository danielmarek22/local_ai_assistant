import time
import logging
from typing import Generator, Optional

from app.core.actions import Action
from app.core.events import AssistantStateEvent
from app.core.assistant_state import AssistantState

logger = logging.getLogger("tool_executor")


class ToolExecutor:
    """
    Executes planner actions that map to tools.
    Handles availability checks, timing, errors, and state events.
    """

    def __init__(self, tools):
        self.tools = tools

    def execute(
        self,
        action: Action,
        user_text: str,
    ) -> Generator[AssistantStateEvent, None, Optional[str]]:
        tool = self.tools.get(action.type)

        if not tool:
            logger.warning(
                "Tool '%s' not registered, skipping",
                action.type,
            )
            return None

        if not tool.is_available:
            logger.warning(
                "Tool '%s' unavailable, skipping",
                action.type,
            )
            return None

        yield AssistantStateEvent(state=AssistantState.SEARCHING)

        logger.info("Running tool '%s'", action.type)
        start_ts = time.perf_counter()

        try:
            query = (action.payload or {}).get("query") or user_text
            logger.debug("Tool '%s' query: %r", action.type, query)

            context = tool.run(query)

            logger.info(
                "Tool '%s' completed (duration=%.2f ms)",
                action.type,
                (time.perf_counter() - start_ts) * 1000,
            )

            if context:
                logger.debug(
                    "Tool '%s' returned context (%d chars)",
                    action.type,
                    len(context),
                )
            else:
                logger.debug(
                    "Tool '%s' returned no context",
                    action.type,
                )

            return context

        except Exception:
            logger.exception(
                "Tool '%s' failed during execution",
                action.type,
            )
            return None
