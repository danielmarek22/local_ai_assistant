import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.core.response_generator import ResponseGenerator
from app.core.events import AssistantSpeechEvent, AvatarAnimationEvent
from app.integrations import ToolResult


def consume(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as completed:
            return events, completed.value


class ResponseGeneratorTests(unittest.TestCase):
    def test_direct_generation_emits_response_without_persistence_dependencies(self):
        trace = Mock()
        generator = ResponseGenerator(
            SimpleNamespace(stream_chat=lambda messages, **kwargs: iter(["Hello [wave]."])),
            SimpleNamespace(), trace, allowed_animations={"wave"},
        )
        events, response = consume(generator.stream_response("session", []))
        self.assertIn("Hello", response)
        self.assertTrue(any(isinstance(event, AvatarAnimationEvent) for event in events))
        self.assertTrue(any(isinstance(event, AssistantSpeechEvent) for event in events))
        trace.assert_not_called()

    def test_native_loop_preserves_authority_and_records_only_tool_trace(self):
        trace = Mock()
        authority = object()
        calls = []
        def execute(call, **kwargs):
            calls.append(kwargs)
            yield from ()
            return ToolResult.success("Observed result")
        llm = SimpleNamespace(chat_buffered=Mock(side_effect=[
            {"content": "", "tool_calls": [{"function": {"name": "demo__run", "arguments": {}}}]},
            {"content": "Finished."},
        ]))
        generator = ResponseGenerator(llm, SimpleNamespace(execute=execute, get_native_tools=lambda **kwargs: []), trace)
        _, response = consume(generator.stream_late_routed_response(
            "session", [], "Do it", authoritative_turn=authority,
        ))
        self.assertEqual(response, "Finished.")
        self.assertIs(calls[0]["authoritative_turn"], authority)
        trace.assert_called_once()
        self.assertEqual(trace.call_args.args[:2], ("session", "system"))

    def test_inference_failure_uses_tool_free_recovery_without_history_access(self):
        trace = Mock()
        llm = SimpleNamespace(chat_buffered=Mock(side_effect=TimeoutError("late")),
                              stream_chat=Mock(return_value=iter(["Recovered."])))
        generator = ResponseGenerator(llm, SimpleNamespace(get_native_tools=lambda **kwargs: []), trace)
        _, response = consume(generator.stream_late_routed_response("session", [], "Question"))
        self.assertEqual(response, "Recovered.")
        self.assertIs(llm.stream_chat.call_args.kwargs["think_override"], False)
        trace.assert_not_called()
