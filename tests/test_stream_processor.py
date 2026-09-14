import unittest

from app.avatar.stream_processor import StreamProcessor


class StreamProcessorTests(unittest.TestCase):
    @staticmethod
    def _text(events):
        return "".join(value for event_type, value in events if event_type == "text")

    def test_push_holds_partial_expression_until_completed(self):
        processor = StreamProcessor()

        self.assertEqual(processor.push("[st"), [])

        events = processor.push("ate:happy]Hello")
        self.assertEqual(events, [("expression", "happy"), ("text", "Hello")])

    def test_flush_releases_trailing_text(self):
        processor = StreamProcessor()

        self.assertEqual(processor.push("Hello "), [("text", "Hello ")])
        self.assertEqual(processor.push("[sta"), [])
        self.assertEqual(processor.flush(), [("text", "[sta")])

    def test_push_emits_animation_for_supported_animation_tag(self):
        processor = StreamProcessor(allowed_animations={"greeting"})

        events = processor.push("[animation:greeting]Hello")
        self.assertEqual(events, [("animation", "greeting"), ("text", "Hello")])

    def test_push_accepts_gesture_alias_for_animation_tag(self):
        processor = StreamProcessor(allowed_animations={"greeting"})

        events = processor.push("[gesture:greeting]Hey")
        self.assertEqual(events, [("animation", "greeting"), ("text", "Hey")])

    def test_push_accepts_bare_animation_name_in_brackets(self):
        processor = StreamProcessor(allowed_animations={"greeting"})

        events = processor.push("[greeting]Hey")
        self.assertEqual(events, [("animation", "greeting"), ("text", "Hey")])

    def test_push_strips_unknown_animation_tags_without_emitting_text(self):
        processor = StreamProcessor(allowed_animations={"greeting"})

        events = processor.push("Hi [animation:unknown]there")
        self.assertEqual(events, [("text", "Hi "), ("text", "there")])

    def test_push_handles_animation_tag_split_across_chunks(self):
        processor = StreamProcessor(allowed_animations={"greeting"})

        self.assertEqual(processor.push("Hello [anim"), [("text", "Hello ")])
        self.assertEqual(
            processor.push("ation:greeting] there"),
            [("animation", "greeting"), ("text", " there")],
        )

    def test_push_preserves_order_for_expression_and_animation_tags(self):
        processor = StreamProcessor(allowed_animations={"greeting"})

        events = processor.push("[state:happy]Hi [animation:greeting]there")
        self.assertEqual(
            events,
            [
                ("expression", "happy"),
                ("text", "Hi "),
                ("animation", "greeting"),
                ("text", "there"),
            ],
        )

    def test_invalid_bracketed_values_are_filtered_fail_closed(self):
        processor = StreamProcessor(allowed_animations={"greeting"})

        events = processor.push(
            "A[system:happy]B[state:unknown]C[animation:other]D[untrusted]E"
        )

        self.assertEqual(self._text(events), "ABCDE")
        self.assertFalse(any(event_type != "text" for event_type, _value in events))

    def test_nested_bracket_candidate_cannot_smuggle_a_control(self):
        processor = StreamProcessor()

        events = processor.push("Before [[state:happy]] after")

        self.assertEqual(self._text(events), "Before  after")
        self.assertFalse(any(event_type == "expression" for event_type, _value in events))

    def test_configured_expression_allowlist_drives_validation(self):
        processor = StreamProcessor(allowed_expressions={"focused"})

        events = processor.push("[state:happy][expression:focused]Ready")

        self.assertEqual(events, [("expression", "focused"), ("text", "Ready")])

    def test_fenced_code_is_preserved_across_chunks_without_controls(self):
        processor = StreamProcessor(allowed_animations={"greeting"})

        events = processor.push("```text\n[state:happy][animation:")
        events.extend(processor.push("greeting]\n``` [state:happy]Visible"))

        self.assertEqual(
            self._text(events),
            "```text\n[state:happy][animation:greeting]\n``` Visible",
        )
        self.assertEqual(
            [value for event_type, value in events if event_type == "expression"],
            ["happy"],
        )
        self.assertFalse(any(event_type == "animation" for event_type, _value in events))

    def test_inline_code_is_preserved_across_chunks_without_controls(self):
        processor = StreamProcessor()

        events = processor.push("Use `[state:")
        events.extend(processor.push("happy]` then [state:sad]continue"))

        self.assertEqual(self._text(events), "Use `[state:happy]` then continue")
        self.assertEqual(
            [value for event_type, value in events if event_type == "expression"],
            ["sad"],
        )

    def test_markdown_links_images_references_and_escapes_are_preserved(self):
        processor = StreamProcessor()
        source = (
            r"\[state:happy] [state:happy](docs) "
            r"![state:sad](face.png) [label][target] [state:relaxed]Done"
        )

        events = processor.push(source)

        self.assertEqual(
            self._text(events),
            r"\[state:happy] [state:happy](docs) "
            r"![state:sad](face.png) [label][target] Done",
        )
        self.assertEqual(
            [value for event_type, value in events if event_type == "expression"],
            ["relaxed"],
        )

    def test_possible_link_is_held_until_following_chunk_disambiguates_it(self):
        processor = StreamProcessor()

        self.assertEqual(processor.push("[state:happy]"), [])
        events = processor.push("(expression-docs)")

        self.assertEqual(self._text(events), "[state:happy](expression-docs)")
        self.assertFalse(any(event_type == "expression" for event_type, _value in events))

    def test_parsing_is_stable_at_every_single_chunk_boundary(self):
        source = (
            r"Start \[literal] `[state:sad]` [invalid] [state:happy] "
            "[docs](guide) ```[animation:greeting]``` "
            "[animation:greeting]End"
        )

        def normalized_events(chunks):
            processor = StreamProcessor(allowed_animations={"greeting"})
            events = []
            for chunk in chunks:
                events.extend(processor.push(chunk))
            events.extend(processor.flush())
            return (
                self._text(events),
                [event for event in events if event[0] != "text"],
            )

        expected = normalized_events([source])
        for split_at in range(len(source) + 1):
            with self.subTest(split_at=split_at):
                self.assertEqual(
                    normalized_events([source[:split_at], source[split_at:]]),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
