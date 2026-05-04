import unittest

from app.core.stream_processor import StreamProcessor


class StreamProcessorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
