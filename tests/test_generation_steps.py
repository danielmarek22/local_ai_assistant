import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.core.events import (
    AssistantSpeechEvent, AssistantStateEvent, AssistantThinkingEvent,
    AvatarAnimationEvent, AvatarExpressionEvent,
)
from app.core.generation_step import AcceptedToolCall, FinalText, InvalidToolCall
from app.core.response_generator import ResponseGenerator


def consume(iterator):
    events = []
    while True:
        try:
            events.append(next(iterator))
        except StopIteration as done:
            return events, done.value


class GenerationStepTests(unittest.TestCase):
    def generator(self, *, chunks=(), messages=()):
        llm = SimpleNamespace(
            resolve_think_value=lambda override: True if override is None else override,
            stream_chat=Mock(side_effect=lambda *args, **kwargs: iter(chunks)),
            chat_buffered=Mock(side_effect=list(messages)),
        )
        executor = SimpleNamespace(get_native_tools=Mock(return_value=[]), execute=Mock())
        return ResponseGenerator(llm, executor, Mock(), allowed_animations={"wave"})

    def test_explicit_expression_and_animation_order_survives_every_chunk_split(self):
        text = "[expression:happy]Hi[wave]"
        expected = [
            AvatarExpressionEvent(expression="happy"),
            AssistantSpeechEvent(text="Hi"), AvatarAnimationEvent(animation="wave"),
        ]
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                generator = self.generator(chunks=[text[:split], text[split:]])
                events, result = consume(generator.stream_response("s", []))
                render_events = expected
                if split == text.index("Hi") + 1:
                    render_events = [expected[0], AssistantSpeechEvent(text="H"),
                                     AssistantSpeechEvent(text="i"), expected[-1]]
                self.assertEqual(events, [AssistantStateEvent(state="responding"), *render_events])
                self.assertEqual(result, "Hi")
        generator = self.generator(messages=[{"content": text}])
        events, result = consume(generator._stream_late_routing_step("s", [], "Hi"))
        self.assertEqual(events, [
            AssistantStateEvent(state="thinking"), AssistantStateEvent(state="responding"), *expected,
        ])
        self.assertEqual(result, FinalText("Hi"))

    def test_empty_and_whitespace_paths_keep_distinct_event_sequences(self):
        for text in ("", "   "):
            with self.subTest(text=text):
                generator = self.generator(chunks=[text], messages=[{"content": text}])
                events, response = consume(generator.stream_response("s", []))
                self.assertEqual(events, [AssistantStateEvent(state="responding"),
                                          AvatarExpressionEvent(expression="neutral")])
                self.assertEqual(response, "")
                events, result = consume(generator._stream_late_routing_step("s", [], "Hi"))
                expected = [AssistantStateEvent(state="thinking")]
                if text:
                    expected += [AssistantStateEvent(state="responding"),
                                 AvatarExpressionEvent(expression="neutral")]
                self.assertEqual(events, expected)
                self.assertEqual(result, FinalText(""))

    def test_animation_only_response_finishes_with_default_expression(self):
        generator = self.generator(chunks=["[wave]"], messages=[{"content": "[wave]"}])
        expected = [AvatarAnimationEvent(animation="wave"), AvatarExpressionEvent(expression="neutral")]
        direct_events, response = consume(generator.stream_response("s", []))
        buffered_events, result = consume(generator._stream_late_routing_step("s", [], "Hi"))
        self.assertEqual(direct_events[1:], expected)
        self.assertEqual(buffered_events[2:], expected)
        self.assertEqual(response, "")
        self.assertEqual(result, FinalText(""))

    def test_unfinished_tag_is_flushed_once_without_duplicate_default_expression(self):
        text = "Hi [unfinished"
        generator = self.generator(chunks=[text], messages=[{"content": text}])
        expected = [AvatarExpressionEvent(expression="neutral"),
                    AssistantSpeechEvent(text="Hi "), AssistantSpeechEvent(text="[unfinished")]
        direct_events, response = consume(generator.stream_response("s", []))
        buffered_events, result = consume(generator._stream_late_routing_step("s", [], "Hi"))
        self.assertEqual(direct_events[1:], expected)
        self.assertEqual(buffered_events[2:], expected)
        self.assertEqual(response, text)
        self.assertEqual(result, FinalText(text))

    def test_thinking_precedes_visible_events_without_entering_returned_text(self):
        generator = self.generator(chunks=["<think>private</think>Hi"],
                                   messages=[{"thinking": "private", "content": "Hi"}])
        events, response = consume(generator.stream_response("s", []))
        self.assertEqual(events, [
            AssistantStateEvent(state="responding"), AssistantThinkingEvent(text="private"),
            AvatarExpressionEvent(expression="neutral"), AssistantSpeechEvent(text="Hi"),
        ])
        self.assertEqual(response, "Hi")
        events, result = consume(generator._stream_late_routing_step("s", [], "Hi"))
        self.assertEqual(events, [
            AssistantStateEvent(state="thinking"), AssistantThinkingEvent(text="private"),
            AssistantStateEvent(state="responding"), AvatarExpressionEvent(expression="neutral"),
            AssistantSpeechEvent(text="Hi"),
        ])
        self.assertEqual(result, FinalText("Hi"))

    def test_first_accepted_tool_suppresses_visible_content_and_later_calls(self):
        arguments = {"command": "pwd"}
        generator = self.generator(messages=[{
            "thinking": "private", "content": "Do not emit [wave]",
            "tool_calls": [
                {"function": {"name": "shell__execute", "arguments": arguments}},
                {"function": {"name": "demo__second", "arguments": {}}},
            ],
        }])
        events, result = consume(generator._stream_late_routing_step("s", [], "Hi"))
        self.assertIsInstance(result, AcceptedToolCall)
        self.assertEqual(result.tool_name, "shell__execute")
        self.assertIs(result.tool_arguments, arguments)
        self.assertEqual(events, [AssistantStateEvent(state="thinking"),
                                  AssistantThinkingEvent(text="private")])
        generator.tool_executor.execute.assert_not_called()

    def test_invalid_call_keeps_raw_name_arguments_and_diagnostic(self):
        for name, arguments, diagnostic in (
            ("beliefs__update", ["not an object"], "Tool arguments must be an object"),
            (123, {"evidence": "original"}, "Invalid capability ID"),
            ("bad name", None, "Invalid capability ID"),
        ):
            with self.subTest(name=name):
                generator = self.generator(messages=[{
                    "content": "Must stay hidden",
                    "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
                }])
                events, result = consume(generator._stream_late_routing_step("s", [], "Hi"))
                self.assertIsInstance(result, InvalidToolCall)
                self.assertEqual(result.tool_name, name)
                self.assertIs(result.tool_arguments, arguments)
                self.assertIn(diagnostic, result.error)
                self.assertEqual(events, [AssistantStateEvent(state="thinking")])

    def test_invalid_belief_arguments_reach_correction_without_execution(self):
        raw_arguments = ["original evidence"]
        generator = self.generator(messages=[
            {"tool_calls": [{"function": {"name": "beliefs__update", "arguments": raw_arguments}}]},
            {"content": "Please clarify."},
        ])
        messages = []
        _, response = consume(generator.stream_late_routed_response("s", messages, "Remember"))
        self.assertEqual(response, "Please clarify.")
        generator.tool_executor.execute.assert_not_called()
        assistant = next(message for message in messages if message["role"] == "assistant")
        self.assertIs(assistant["tool_calls"][0]["function"]["arguments"], raw_arguments)
        observation = next(message for message in messages if message["role"] == "tool")
        self.assertIn("Tool arguments must be an object", observation["content"])
        calls = generator.llm.chat_buffered.call_args_list
        self.assertEqual([call.kwargs["generation_phase"] for call in calls], ["initial", "correction"])
        self.assertIs(calls[1].kwargs["think_override"], False)


if __name__ == "__main__":
    unittest.main()
