import json
import unittest
from unittest.mock import Mock, patch

import requests

from app.llm.base import InferenceFailure
from app.llm.ollama_stream import OllamaClient


class OllamaStreamDecodingTests(unittest.TestCase):
    def client(self, lines):
        response = requests.Response()
        response.status_code = 200
        response._content = b"\n".join(lines)
        response._content_consumed = True
        client = OllamaClient(model="test", host="http://localhost:11434")
        self.addCleanup(client.close)
        client.session.post = Mock(return_value=response)
        return client

    def test_blank_lines_and_terminal_metrics_preserve_both_modes(self):
        lines = [
            b"", b'{"message":{"thinking":"consider"},"done":false}',
            b'{"message":{"content":"answer"},"done":false}',
            b"", b'{"message":{},"done":true,"eval_count":3,"prompt_eval_count":2}',
        ]
        for buffered in (False, True):
            with self.subTest(buffered=buffered):
                client = self.client(lines)
                client.telemetry = Mock(enabled=True, full=False, extended=False)
                if buffered:
                    self.assertEqual(client.chat_buffered([]),
                                     {"content": "answer", "thinking": "consider"})
                else:
                    self.assertEqual(list(client.stream_chat([])),
                                     ["<think>\n", "consider", "\n</think>\n\n", "answer"])
                metrics = client.telemetry.record_model_call.call_args.kwargs
                self.assertEqual(metrics["prompt_tokens"], 2)
                self.assertEqual(metrics["completion_tokens"], 3)
                client.session.post.assert_called_once()

    def test_decoding_errors_keep_mode_specific_exception_types(self):
        for line, native_error in ((b"not json", json.JSONDecodeError),
                                   (b"\xff", UnicodeDecodeError)):
            for buffered in (False, True):
                with self.subTest(line=line, buffered=buffered):
                    client = self.client([line])
                    expected = InferenceFailure if buffered else native_error
                    with self.assertRaises(expected) as raised:
                        if buffered:
                            client.chat_buffered([])
                        else:
                            list(client.stream_chat([]))
                    if buffered:
                        self.assertEqual(raised.exception.category, "malformed_response")
                        self.assertIsInstance(raised.exception.__cause__, native_error)
                    client.session.post.assert_called_once()

    def test_buffered_rejects_invalid_shapes_and_fields(self):
        invalid_chunks = [
            [], {"message": []}, {"message": None},
            {"message": {"content": 123}}, {"message": {"thinking": True}},
            {"message": {"tool_calls": {"function": {}}}},
        ]
        for chunk in invalid_chunks:
            with self.subTest(chunk=chunk):
                client = self.client([json.dumps(chunk).encode(), b'{"done":true}'])
                with self.assertRaises(InferenceFailure) as raised:
                    client.chat_buffered([])
                self.assertEqual(raised.exception.category, "malformed_response")
                client.session.post.assert_called_once()

    def test_buffered_line_limit_does_not_apply_to_visible_stream(self):
        line = b'{"message":{"content":"answer"},"done":true}'
        with patch("app.llm.stream_decoder.MAX_BUFFERED_LINE_BYTES", len(line) - 1):
            client = self.client([line])
            with self.assertRaisesRegex(InferenceFailure, "bounded line size"):
                client.chat_buffered([])
            self.assertEqual(list(self.client([line]).stream_chat([])), ["answer"])
        with patch("app.llm.stream_decoder.MAX_BUFFERED_LINE_BYTES", len(line)):
            self.assertEqual(self.client([line]).chat_buffered([])["content"], "answer")

    def test_missing_completion_marker_remains_mode_specific(self):
        lines = [b'{"message":{"content":"partial"},"done":false}']
        self.assertEqual(list(self.client(lines).stream_chat([])), ["partial"])
        with self.assertRaisesRegex(InferenceFailure, "completed generation marker"):
            self.client(lines).chat_buffered([])

    def test_generation_deadline_stops_each_mode_without_replay(self):
        for buffered in (False, True):
            with self.subTest(buffered=buffered):
                client = self.client([b'{"message":{"content":"late"},"done":true}'])
                with patch("app.llm.ollama_stream.time.perf_counter", side_effect=[0.0, 2.0]):
                    with self.assertRaises(InferenceFailure) as raised:
                        if buffered:
                            client.chat_buffered([], generation_deadline_s=1.0)
                        else:
                            list(client.stream_chat([], generation_deadline_s=1.0))
                self.assertEqual(raised.exception.category, "generation_deadline")
                client.session.post.assert_called_once()

    def test_buffered_image_error_does_not_start_fallback(self):
        client = self.client([])
        response = client.session.post.return_value
        response.status_code = 400
        response._content = b'{"error":"invalid image data"}'
        with self.assertRaises(InferenceFailure):
            client.chat_buffered([{"role": "user", "images": ["one"]}])
        client.session.post.assert_called_once()
        self.assertTrue(client._multimodal_supported)


if __name__ == "__main__":
    unittest.main()
