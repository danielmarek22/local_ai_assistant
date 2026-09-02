import unittest

from app.core.assistant_state import AssistantState
from app.core.events import AssistantStateEvent, AvatarOutfitEvent
from app.integrations import (
    ApprovalRequest,
    AvatarOutfitEffect,
    CapabilityId,
    IntegrationRegistry,
    RegisteredTool,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from app.services.tool_executor import ToolExecutor


def consume_generator(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


class FakeIntegration:
    name = "demo"

    def __init__(self, available=True, raises=False, use_approval=False):
        self.available = available
        self.raises = raises
        self.use_approval = use_approval
        self.calls = []

    def registered_tools(self):
        return [RegisteredTool(
            spec=ToolSpec(
                capability=CapabilityId("demo", "run"),
                description="Run demo.",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            ),
            handler=self._run,
            available=lambda: self.available,
        )]

    def _run(self, arguments, context):
        self.calls.append((dict(arguments), context))
        if self.raises:
            raise RuntimeError("boom")
        if self.use_approval:
            approved = context.approval_callback(ApprovalRequest(
                capability=CapabilityId("demo", "run"),
                title="Approve demo?",
                reason="Demo requires approval.",
                detail_label="Value",
                detail=str(arguments["value"]),
            ))
            if not approved:
                return ToolResult.denied("Denied")
        return ToolResult.success("context")

    def context(self, _invocation):
        return None


class ToolExecutorTests(unittest.TestCase):
    def _executor(self, integration=None):
        return ToolExecutor(IntegrationRegistry([integration] if integration else []))

    def test_unknown_capability_returns_typed_error_after_state_event(self):
        executor = self._executor()
        call = ToolCall(CapabilityId("missing", "run"), {})

        events, result = consume_generator(
            executor.execute(call, "session-1", "user text")
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].state, AssistantState.SEARCHING)
        self.assertEqual(result.status.value, "error")

    def test_unavailable_capability_returns_typed_result(self):
        integration = FakeIntegration(available=False)
        executor = self._executor(integration)
        call = ToolCall(CapabilityId("demo", "run"), {"value": "x"})

        _events, result = consume_generator(
            executor.execute(call, "session-1", "user text")
        )

        self.assertEqual(result.status.value, "unavailable")
        self.assertEqual(integration.calls, [])

    def test_valid_call_yields_searching_and_preserves_arguments(self):
        integration = FakeIntegration()
        executor = self._executor(integration)
        call = ToolCall(CapabilityId("demo", "run"), {"value": "my value"})

        events, result = consume_generator(
            executor.execute(call, "session-1", "user text")
        )

        self.assertIsInstance(events[0], AssistantStateEvent)
        self.assertEqual(events[0].state, AssistantState.SEARCHING)
        self.assertEqual(result.content, "context")
        self.assertEqual(integration.calls[0][0], {"value": "my value"})
        self.assertEqual(integration.calls[0][1].session_id, "session-1")

    def test_schema_error_does_not_fall_back_to_user_text(self):
        integration = FakeIntegration()
        executor = self._executor(integration)
        call = ToolCall(CapabilityId("demo", "run"), {})

        _events, result = consume_generator(
            executor.execute(call, "session-1", "fallback text")
        )

        self.assertEqual(result.status.value, "error")
        self.assertEqual(integration.calls, [])

    def test_handler_exception_becomes_typed_error(self):
        executor = self._executor(FakeIntegration(raises=True))
        call = ToolCall(CapabilityId("demo", "run"), {"value": "x"})

        _events, result = consume_generator(
            executor.execute(call, "session-1", "user text")
        )

        self.assertEqual(result.status.value, "error")
        self.assertIn("execution failed", result.content)

    def test_approval_request_is_serialized_for_transport_callback(self):
        executor = self._executor(FakeIntegration(use_approval=True))
        call = ToolCall(CapabilityId("demo", "run"), {"value": "x"})
        requests = []

        def approve(request):
            requests.append(request)
            return True

        _events, result = consume_generator(
            executor.execute(
                call,
                "session-1",
                "user text",
                approval_callback=approve,
            )
        )

        self.assertEqual(result.status.value, "success")
        self.assertEqual(requests[0]["tool"], "demo__run")
        self.assertEqual(requests[0]["detail_label"], "Value")

    def test_typed_outfit_effect_becomes_avatar_event(self):
        integration = FakeIntegration()
        integration._run = lambda _arguments, _context: ToolResult.success(
            "changed",
            effects=(AvatarOutfitEffect("pajamas", "/static/avatars/pajamas.vrm"),),
        )
        executor = self._executor(integration)

        events, result = consume_generator(executor.execute(
            ToolCall(CapabilityId("demo", "run"), {"value": "x"}),
            "session-1",
            "change",
        ))

        self.assertEqual(result.status.value, "success")
        outfit_event = next(event for event in events if isinstance(event, AvatarOutfitEvent))
        self.assertEqual(outfit_event.outfit, "pajamas")


if __name__ == "__main__":
    unittest.main()
