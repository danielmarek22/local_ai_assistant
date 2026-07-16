import time
import logging
from typing import Generator, Optional

from app.core.actions import Action
from app.core.events import AssistantStateEvent
from app.core.assistant_state import AssistantState
from app.logging import trace_event

logger = logging.getLogger("tool_executor")


class ToolExecutor:
    """
    Executes actions that map to tools.
    Handles availability checks, timing, errors, and state events.
    """

    def __init__(self, tools):
        self.tools = tools

    def get_native_tools(self) -> list[dict]:
            """
            Dynamically extracts available tool metadata and formats it
            into Ollama-compatible JSON schemas.
            """
            native_tools = []
            for name, tool in self.tools.items():
                if getattr(tool, "is_available", False):
                    native_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": getattr(tool, "description", "No description provided."),
                            "parameters": getattr(tool, "parameters", {"type": "object", "properties": {}})
                        }
                    })
            return native_tools

    def execute(
        self,
        action: Action,
        user_text: str,
    ) -> Generator[AssistantStateEvent, None, Optional[str]]:
        # Use .value for safe dictionary lookup
        tool = self.tools.get(action.type.value)

        if not tool:
            logger.warning(
                "Tool '%s' not registered, skipping",
                action.type.value,
            )
            return None

        if getattr(tool, "is_available", False) is False:
            logger.warning(
                "Tool '%s' unavailable, skipping",
                action.type.value,
            )
            return None

        yield AssistantStateEvent(state=AssistantState.SEARCHING)

        logger.info("Running tool '%s'", action.type.value)
        start_ts = time.perf_counter()

        try:
            # Preserve structured payloads for native tools like write_memory,
            # while keeping the existing string-friendly behavior for search/bash tools.
            payload = action.payload or {}
            if action.type.value == "write_memory" and isinstance(payload, dict):
                primary_arg = payload
            else:
                primary_arg = payload.get("command") or payload.get("query") or user_text

            trace_event(
                "tool_executor",
                "tool_call",
                payload={
                    "tool": action.type.value,
                    "action_payload": action.payload,
                    "primary_arg": primary_arg,
                },
            )

            context = tool.run(primary_arg)

            logger.info(
                "Tool '%s' completed (duration=%.2f ms)",
                action.type.value,
                (time.perf_counter() - start_ts) * 1000,
            )

            if context:
                trace_event(
                    "tool_executor",
                    "tool_result",
                    payload={
                        "tool": action.type.value,
                        "context": context,
                    },
                )
            else:
                logger.debug(
                    "Tool '%s' returned no context",
                    action.type.value,
                )

            return context

        except Exception:
            logger.exception(
                "Tool '%s' failed during execution",
                action.type.value,
            )
            return None