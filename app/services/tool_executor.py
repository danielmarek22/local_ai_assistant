import logging
import time
from collections.abc import Callable, Generator

from app.core.assistant_state import AssistantState
from app.core.events import AssistantStateEvent
from app.integrations import (
    ApprovalRequest,
    IntegrationRegistry,
    InvocationContext,
    ToolCall,
    ToolResult,
)
from app.logging import trace_event


logger = logging.getLogger("tool_executor")


class ToolExecutor:
    """Adds assistant events and tracing around registry capability execution."""

    def __init__(self, registry: IntegrationRegistry):
        self.registry = registry

    def get_native_tools(self) -> list[dict]:
        return self.registry.get_native_tools()

    def collect_context(self, session_id: str, user_text: str, max_chars: int) -> str | None:
        return self.registry.collect_context(
            InvocationContext(session_id=session_id, user_text=user_text),
            max_chars=max_chars,
        )

    def execute(
        self,
        call: ToolCall,
        session_id: str,
        user_text: str,
        approval_callback: Callable[[dict], bool] | None = None,
    ) -> Generator[AssistantStateEvent, None, ToolResult]:
        yield AssistantStateEvent(state=AssistantState.SEARCHING)
        capability = str(call.capability)
        logger.info("Running capability '%s'", capability)
        start_ts = time.perf_counter()

        typed_approval_callback = None
        if approval_callback is not None:
            def typed_approval_callback(request: ApprovalRequest) -> bool:
                return approval_callback(request.to_payload())

        trace_event(
            "tool_executor",
            "tool_call",
            session_id=session_id,
            payload={
                "tool": capability,
                "arguments": dict(call.arguments),
            },
        )

        result = self.registry.invoke(
            call,
            InvocationContext(
                session_id=session_id,
                user_text=user_text,
                approval_callback=typed_approval_callback,
            ),
        )

        logger.info(
            "Capability '%s' completed (status=%s, duration=%.2f ms)",
            capability,
            result.status.value,
            (time.perf_counter() - start_ts) * 1000,
        )
        trace_event(
            "tool_executor",
            "tool_result",
            session_id=session_id,
            payload={
                "tool": capability,
                "status": result.status.value,
                "content": result.content,
            },
        )
        return result
