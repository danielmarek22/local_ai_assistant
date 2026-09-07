import logging
import time
import uuid
from collections.abc import Callable, Generator

from app.core.assistant_state import AssistantState
from app.core.events import AssistantStateEvent, AvatarOutfitEvent
from app.integrations import (
    ApprovalRequest,
    AvatarOutfitEffect,
    CapabilityId,
    IntegrationRegistry,
    InvocationContext,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    NotificationRequest,
)
from app.logging import trace_event


logger = logging.getLogger("tool_executor")


class ToolExecutor:
    """Adds assistant events and tracing around registry capability execution."""

    def __init__(self, registry: IntegrationRegistry, operation_store=None):
        self.registry = registry
        self.operation_store = operation_store

    def get_native_tools(
        self,
        allowed_capabilities: set[CapabilityId] | None = None,
        *,
        session_id: str = "",
        user_text: str = "",
        authoritative_turn=None,
        prepared_belief_turn=None,
    ) -> list[dict]:
        return self.registry.get_native_tools(
            allowed_capabilities,
            InvocationContext(
                session_id=session_id,
                user_text=user_text,
                authoritative_turn=authoritative_turn,
                prepared_belief_turn=prepared_belief_turn,
            ),
        )

    def collect_context(self, session_id: str, user_text: str, max_chars: int) -> str | None:
        return self.registry.collect_context(
            InvocationContext(session_id=session_id, user_text=user_text),
            max_chars=max_chars,
        )

    def close(self) -> None:
        self.registry.close()

    def execute(
        self,
        call: ToolCall,
        session_id: str,
        user_text: str,
        approval_callback: Callable[[dict], bool] | None = None,
        event_id: str | None = None,
        root_event_id: str | None = None,
        causation_id: str | None = None,
        notification_callback: Callable[[NotificationRequest], bool] | None = None,
        authoritative_turn=None,
        prepared_belief_turn=None,
    ) -> Generator[AssistantStateEvent, None, ToolResult]:
        yield AssistantStateEvent(state=AssistantState.SEARCHING)
        capability = str(call.capability)
        logger.info("Running capability '%s'", capability)
        start_ts = time.perf_counter()
        invocation_id = str(uuid.uuid4())

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

        if self.operation_store is not None:
            self.operation_store.begin_operation(
                invocation_id=invocation_id,
                capability=capability,
                session_id=session_id,
                event_id=event_id,
                root_event_id=root_event_id,
                causation_id=causation_id,
            )

        result = self.registry.invoke(
            call,
            InvocationContext(
                session_id=session_id,
                user_text=user_text,
                approval_callback=typed_approval_callback,
                invocation_id=invocation_id,
                event_id=event_id,
                root_event_id=root_event_id,
                causation_id=causation_id,
                notification_callback=notification_callback,
                authoritative_turn=authoritative_turn,
                prepared_belief_turn=prepared_belief_turn,
            ),
        )
        if result.status == ToolResultStatus.PENDING and result.operation_id != invocation_id:
            logger.error("Capability %s returned an unexpected operation ID", capability)
            result = ToolResult.error(
                f"Capability returned an invalid asynchronous operation ID: {capability}"
            )
        if self.operation_store is not None:
            self.operation_store.finish_operation(
                invocation_id,
                result.status.value,
                result.content,
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
        for effect in result.effects:
            if isinstance(effect, AvatarOutfitEffect):
                yield AvatarOutfitEvent(outfit=effect.outfit, url=effect.url)
        return result
