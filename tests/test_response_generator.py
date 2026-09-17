import unittest
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

from app.core.response_generator import ResponseGenerator
from app.core.events import AssistantSpeechEvent, AvatarAnimationEvent
from app.integrations import ToolResult
from app.core.tool_executor import ToolExecutor
from app.llm.base import LLMClient


def consume(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as completed:
            return events, completed.value


class ResponseGeneratorTests(unittest.TestCase):
    def test_stream_controls_reach_model_without_signature_probing(self):
        llm = create_autospec(LLMClient, instance=True, spec_set=True)
        llm.stream_chat.return_value = iter(["Hello."])
        messages = [{"role": "user", "content": "Hi"}]
        _, response = consume(ResponseGenerator(llm, Mock(), Mock()).stream_response(
            "session", messages, think_override=False,
            options_override={"num_predict": 64}, generation_deadline_s=30.0,
            timeout_override=10.0,
        ))
        self.assertEqual(response, "Hello.")
        llm.stream_chat.assert_called_once_with(
            messages, think_override=False, options_override={"num_predict": 64},
            generation_deadline_s=30.0, timeout_override=10.0,
            telemetry_session_id="session",
        )

    def test_buffered_generation_receives_controls_and_discovery_context(self):
        llm = create_autospec(LLMClient, instance=True, spec_set=True)
        llm.resolve_think_value.return_value = "low"
        llm.chat_buffered.return_value = {"content": "Finished."}
        executor = create_autospec(ToolExecutor, instance=True, spec_set=True)
        executor.get_native_tools.return_value = []
        authority = object()
        prepared = Mock()
        prepared.tool_catalog_message.return_value = "Frozen belief catalog"
        messages = []
        generator = ResponseGenerator(llm, executor, Mock(), generation_deadline_s=42.0)
        _, response = consume(generator.stream_late_routed_response(
            "session", messages, "Question", allowed_capabilities=frozenset(),
            authoritative_turn=authority, prepared_belief_turn=prepared,
            initial_think_override="low",
        ))
        self.assertEqual(response, "Finished.")
        llm.resolve_think_value.assert_called_once_with("low")
        executor.get_native_tools.assert_called_once_with(
            allowed_capabilities=frozenset(), session_id="session", user_text="Question",
            authoritative_turn=authority, prepared_belief_turn=prepared,
        )
        llm.chat_buffered.assert_called_once_with(
            messages=messages, think_override="low", options_override=None, tools=[],
            timeout_override=42.0, generation_deadline_s=42.0,
            generation_phase="initial", react_iteration=1, telemetry_session_id="session",
        )
        llm.chat.assert_not_called()

    def test_internal_discovery_type_error_is_not_retried(self):
        llm = create_autospec(LLMClient, instance=True, spec_set=True)
        llm.resolve_think_value.return_value = False
        executor = create_autospec(ToolExecutor, instance=True, spec_set=True)
        error = TypeError("internal discovery failure")
        executor.get_native_tools.side_effect = error
        generator = ResponseGenerator(llm, executor, Mock())
        with self.assertRaises(TypeError) as raised:
            consume(generator.stream_late_routed_response("session", [], "Question"))
        self.assertIs(raised.exception, error)
        executor.get_native_tools.assert_called_once()
        llm.chat_buffered.assert_not_called()
        llm.chat.assert_not_called()
        executor.execute.assert_not_called()

    def test_omitted_memory_call_does_not_add_a_verification_inference(self):
        llm = SimpleNamespace(
            resolve_think_value=lambda override: True if override is None else override,
            chat_buffered=Mock(side_effect=[
                {"content": "I'll remember that."},
            ]),
        )
        executor = SimpleNamespace(
            get_native_tools=lambda **kwargs: [{"type": "function", "function": {
                "name": "memory__write", "parameters": {"type": "object"},
            }}],
            execute=Mock(),
        )
        _, response = consume(ResponseGenerator(llm, executor, Mock()).stream_late_routed_response(
            "session", [], "Remember our project decision.",
        ))
        self.assertEqual(response, "I'll remember that.")
        llm.chat_buffered.assert_called_once()
        executor.execute.assert_not_called()

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
        llm = SimpleNamespace(
            resolve_think_value=lambda override: True if override is None else override,
            chat_buffered=Mock(side_effect=[
                {"content": "", "tool_calls": [{"function": {"name": "demo__run", "arguments": {}}}]},
                {"content": "Finished."},
            ]),
        )
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
        llm = SimpleNamespace(
            resolve_think_value=lambda override: True if override is None else override,
            chat_buffered=Mock(side_effect=TimeoutError("late")),
            stream_chat=Mock(return_value=iter(["Recovered."])),
        )
        generator = ResponseGenerator(llm, SimpleNamespace(get_native_tools=lambda **kwargs: []), trace)
        _, response = consume(generator.stream_late_routed_response("session", [], "Question"))
        self.assertEqual(response, "Recovered.")
        self.assertIs(llm.stream_chat.call_args.kwargs["think_override"], False)
        self.assertEqual(llm.stream_chat.call_args.kwargs["options_override"], {"num_predict": 192})
        self.assertEqual(llm.stream_chat.call_args.kwargs["generation_deadline_s"], 180.0)
        self.assertEqual(llm.stream_chat.call_args.kwargs["timeout_override"], 180.0)
        self.assertEqual(llm.stream_chat.call_args.kwargs["telemetry_session_id"], "session")
        trace.assert_not_called()
