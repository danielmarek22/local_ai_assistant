import copy
import json
import unittest
from unittest.mock import Mock

import requests

from app.llm.base import InferenceFailure
from app.llm.ollama_stream import OllamaClient


def response(mode, *, status=200, error="invalid image data"):
    result = requests.Response()
    result.status_code = status
    data = {"error": error} if status != 200 else {"message": {"content": "ok"}, "done": True}
    result._content = (json.dumps(data) + ("\n" if mode == "stream" else "")).encode()
    result._content_consumed = True
    return result


class OllamaImageFallbackTests(unittest.TestCase):
    def client(self, mode, replies):
        client = OllamaClient(model="test", host="http://localhost:11434", max_retries=0)
        self.addCleanup(client.close)
        payloads = []
        replies = iter(replies)

        def post(*args, **kwargs):
            payloads.append(copy.deepcopy(kwargs["json"]))
            item = next(replies)
            if isinstance(item, Exception):
                raise item
            return item

        client.session.post = Mock(side_effect=post)
        return client, payloads

    def invoke(self, client, mode, messages):
        if mode == "chat":
            return client.chat(messages)["content"]
        return "".join(client.stream_chat(messages))

    def test_candidate_order_is_shared_and_success_metadata_resets(self):
        messages = [{"role": "user", "content": "look", "images": ["one", "two"]}]
        original = copy.deepcopy(messages)
        for mode in ("chat", "stream"):
            with self.subTest(mode=mode):
                client, payloads = self.client(mode, [
                    response(mode, status=400), response(mode, status=422),
                    response(mode), response(mode),
                ])
                self.assertEqual(self.invoke(client, mode, messages), "ok")
                self.assertEqual(
                    [item["messages"][0]["images"] for item in payloads],
                    [["one", "two"], ["two"], ["one"]],
                )
                self.assertEqual(getattr(client, f"last_{mode}_image_fallback_strategy"),
                                 "without image 2 from message 1")
                self.assertEqual(getattr(client, f"last_{mode}_dropped_current_images_count"), 1)
                self.assertTrue(getattr(client, f"last_{mode}_dropped_current_images"))
                self.assertEqual(self.invoke(client, mode, messages), "ok")
                self.assertIsNone(getattr(client, f"last_{mode}_image_fallback_strategy"))
                self.assertEqual(getattr(client, f"last_{mode}_dropped_current_images_count"), 0)
                self.assertFalse(getattr(client, f"last_{mode}_dropped_current_images"))
                self.assertEqual(messages, original)

    def test_dropping_only_historical_images_does_not_report_current_image_loss(self):
        messages = [
            {"role": "user", "content": "old", "images": ["old"]},
            {"role": "user", "content": "new", "images": ["new"]},
        ]
        for mode in ("chat", "stream"):
            with self.subTest(mode=mode):
                client, payloads = self.client(mode, [
                    response(mode, status=400), response(mode, status=400), response(mode),
                ])
                self.assertEqual(self.invoke(client, mode, messages), "ok")
                self.assertNotIn("images", payloads[-1]["messages"][0])
                self.assertEqual(payloads[-1]["messages"][1]["images"], ["new"])
                self.assertEqual(getattr(client, f"last_{mode}_dropped_current_images_count"), 0)
                self.assertFalse(getattr(client, f"last_{mode}_dropped_current_images"))

    def test_exhausted_candidates_fail_without_success_metadata(self):
        messages = [{"role": "user", "content": "look", "images": ["one", "two"]}]
        for mode in ("chat", "stream"):
            with self.subTest(mode=mode):
                replies = [response(mode, status=400) for _ in range(4)]
                client, payloads = self.client(mode, replies)
                with self.assertRaises(InferenceFailure) as raised:
                    self.invoke(client, mode, messages)
                self.assertIs(raised.exception.__cause__.response, replies[-1])
                self.assertEqual(len(payloads), 4)
                self.assertNotIn("images", payloads[-1]["messages"][0])
                self.assertIsNone(getattr(client, f"last_{mode}_image_fallback_strategy"))
                self.assertFalse(getattr(client, f"last_{mode}_dropped_current_images"))

    def test_unsupported_model_remains_cached_on_next_request(self):
        messages = [{"role": "user", "content": "look", "images": ["one"]}]
        for mode in ("chat", "stream"):
            with self.subTest(mode=mode):
                client, payloads = self.client(mode, [
                    response(mode, status=400, error="model does not support images"),
                    response(mode), response(mode),
                ])
                self.assertEqual(self.invoke(client, mode, messages), "ok")
                self.assertEqual(self.invoke(client, mode, messages), "ok")
                self.assertEqual(len(payloads), 3)
                self.assertNotIn("images", payloads[-1]["messages"][0])
                self.assertIsNone(getattr(client, f"last_{mode}_image_fallback_strategy"))

    def test_partial_stream_is_not_replayed_or_marked_as_successful_fallback(self):
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                interrupted = response("stream")
                error = requests.ReadTimeout("interrupted")

                def lines():
                    yield b'{"message":{"content":"partial"},"done":false}'
                    raise error

                interrupted.iter_lines = lines
                replies = [response("stream", status=400)] if fallback else []
                client, payloads = self.client("stream", replies + [interrupted])
                stream = client.stream_chat([{"role": "user", "images": ["one"]}])
                self.assertEqual(next(stream), "partial")
                # Preserve the existing exception types of both paths.
                expected = requests.ReadTimeout if fallback else InferenceFailure
                with self.assertRaises(expected):
                    next(stream)
                self.assertEqual(len(payloads), 2 if fallback else 1)
                self.assertIsNone(client.last_stream_image_fallback_strategy)
                self.assertFalse(client.last_stream_dropped_current_images)


if __name__ == "__main__":
    unittest.main()
