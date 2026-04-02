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


if __name__ == "__main__":
    unittest.main()
