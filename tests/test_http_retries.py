import unittest
from unittest.mock import patch

import requests

from app.llm.ollama_stream import OllamaClient
from app.tools.web_search import SearXNGClient


class _FakeResponse:
    def __init__(self, data=None, status_code=200, http_error=None):
        self._data = data or {}
        self.status_code = status_code
        self._http_error = http_error

    def raise_for_status(self):
        if self._http_error is not None:
            raise self._http_error

    def json(self):
        return self._data


class _FakeStreamResponse(_FakeResponse):
    def __init__(self, lines, status_code=200, http_error=None):
        super().__init__(data=None, status_code=status_code, http_error=http_error)
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        for line in self._lines:
            yield line


def _ndjson_line(payload: str) -> bytes:
    return payload.encode("utf-8")


class HttpRetryTests(unittest.TestCase):
    @patch("app.llm.ollama_stream.time.sleep", return_value=None)
    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_chat_retries_on_timeout_then_succeeds(self, post_mock, _sleep_mock):
        post_mock.side_effect = [
            requests.Timeout("timeout"),
            _FakeResponse(
                data={"message": {"content": "ok"}, "done_reason": "stop"},
            ),
        ]

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
            timeout_s=0.1,
            max_retries=2,
            retry_backoff_s=0.0,
        )

        result = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(result, "ok")
        self.assertEqual(post_mock.call_count, 2)
        self.assertEqual(post_mock.call_args.kwargs["json"]["think"], False)

    @patch("app.llm.ollama_stream.time.sleep", return_value=None)
    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_chat_does_not_retry_on_http_400(self, post_mock, _sleep_mock):
        response = requests.Response()
        response.status_code = 400
        post_mock.return_value = _FakeResponse(
            http_error=requests.HTTPError(response=response),
            status_code=400,
        )

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
            timeout_s=0.1,
            max_retries=3,
            retry_backoff_s=0.0,
        )

        with self.assertRaises(requests.HTTPError):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(post_mock.call_count, 1)

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_chat_retries_without_images_on_http_500(self, post_mock):
        error_response = requests.Response()
        error_response.status_code = 500
        error_response._content = b'{"error":"internal server error"}'

        post_mock.side_effect = [
            _FakeResponse(
                http_error=requests.HTTPError(response=error_response),
                status_code=500,
            ),
            _FakeResponse(
                data={"message": {"content": "text fallback"}, "done_reason": "stop"},
            ),
        ]

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
            max_retries=2,
            retry_backoff_s=0.0,
        )

        result = client.chat(
            [{"role": "user", "content": "what is this?", "images": ["aGVsbG8="]}],
            think_override=False,
        )

        self.assertEqual(result, "text fallback")
        self.assertEqual(post_mock.call_count, 2)
        payload = post_mock.call_args.kwargs["json"]
        self.assertNotIn("images", payload["messages"][0])
        self.assertTrue(client.last_chat_dropped_current_images)
        self.assertEqual(client.last_chat_dropped_current_images_count, 1)

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_chat_uses_native_endpoint_for_images(self, post_mock):
        post_mock.return_value = _FakeResponse(
            data={"message": {"content": "image summary"}, "done_reason": "stop"},
        )

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
        )

        result = client.chat(
            [{"role": "user", "content": "what is this?", "images": ["aGVsbG8="]}],
            think_override=False,
            options_override={"num_predict": 160},
        )

        self.assertEqual(result, "image summary")
        self.assertEqual(post_mock.call_args.args[0], "http://localhost:11434/api/chat")
        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0]["images"], ["aGVsbG8="])
        self.assertEqual(payload["messages"][0]["content"], "what is this?")
        self.assertEqual(payload["think"], False)
        self.assertEqual(payload["options"]["num_predict"], 160)

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_uses_configured_thinking_level(self, post_mock):
        post_mock.return_value = _FakeResponse(
            data={"message": {"content": "ok"}, "done_reason": "stop"},
        )

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
            thinking_enabled=True,
            thinking_level="medium",
        )

        client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(post_mock.call_args.kwargs["json"]["think"], "medium")

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_surfaces_thinking_when_content_is_empty(self, post_mock):
        post_mock.return_value = _FakeStreamResponse(
            lines=[
                _ndjson_line(
                    '{"message":{"thinking":"Thinking Process:\\n\\n1. Hello"},"done":false}'
                ),
                _ndjson_line('{"message":{},"done":true,"done_reason":"stop"}'),
            ]
        )

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
        )

        chunks = list(client.stream_chat([{"role": "user", "content": "hello"}], think_override=True))

        self.assertEqual(
            chunks,
            ["<think>\n", "Thinking Process:\n\n1. Hello", "\n</think>\n\n"],
        )

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_passes_inline_thinking_content_through(self, post_mock):
        post_mock.return_value = _FakeStreamResponse(
            lines=[
                _ndjson_line(
                    '{"message":{"content":"<think>secret</think>Visible reply"},"done":false}'
                ),
                _ndjson_line('{"message":{},"done":true,"done_reason":"stop"}'),
            ]
        )

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
        )

        chunks = list(client.stream_chat([{"role": "user", "content": "hello"}]))

        self.assertEqual(
            chunks,
            ["<think>secret</think>Visible reply"],
        )

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_passes_split_inline_thinking_content_through(self, post_mock):
        post_mock.return_value = _FakeStreamResponse(
            lines=[
                _ndjson_line('{"message":{"content":"<thi"},"done":false}'),
                _ndjson_line('{"message":{"content":"nk>secret</th"},"done":false}'),
                _ndjson_line('{"message":{"content":"ink>Visible"},"done":false}'),
                _ndjson_line('{"message":{},"done":true,"done_reason":"stop"}'),
            ]
        )

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
        )

        chunks = list(client.stream_chat([{"role": "user", "content": "hello"}]))

        self.assertEqual(
            chunks,
            ["<thi", "nk>secret</th", "ink>Visible"],
        )

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_uses_configured_repeat_penalty(self, post_mock):
        post_mock.return_value = _FakeStreamResponse(
            lines=[
                _ndjson_line('{"message":{"content":"ok"},"done":false}'),
                _ndjson_line('{"message":{},"done":true,"done_reason":"stop"}'),
            ]
        )

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
            options={"repeat_penalty": 1.33},
        )

        chunks = list(client.stream_chat([{"role": "user", "content": "hello"}]))

        self.assertEqual(chunks, ["ok"])
        self.assertEqual(post_mock.call_args.kwargs["json"]["options"]["repeat_penalty"], 1.33)

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_applies_thinking_option_overrides(self, post_mock):
        post_mock.return_value = _FakeStreamResponse(
            lines=[
                _ndjson_line('{"message":{"content":"ok"},"done":false}'),
                _ndjson_line('{"message":{},"done":true,"done_reason":"stop"}'),
            ]
        )

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
            options={
                "temperature": 1.0,
                "top_p": 0.95,
                "repeat_penalty": 1.15,
                "min_p": 0.05,
            },
            thinking_options={
                "temperature": 0.6,
                "repeat_penalty": 1.0,
                "min_p": None,
            },
        )

        chunks = list(client.stream_chat([{"role": "user", "content": "hello"}], think_override=True))

        self.assertEqual(chunks, ["ok"])
        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["temperature"], 0.6)
        self.assertEqual(payload["options"]["repeat_penalty"], 1.0)
        self.assertEqual(payload["options"]["top_p"], 0.95)
        self.assertNotIn("min_p", payload["options"])

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_keeps_normal_options_when_thinking_is_disabled(self, post_mock):
        post_mock.return_value = _FakeStreamResponse(
            lines=[
                _ndjson_line('{"message":{"content":"ok"},"done":false}'),
                _ndjson_line('{"message":{},"done":true,"done_reason":"stop"}'),
            ]
        )

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
            options={
                "temperature": 1.0,
                "repeat_penalty": 1.15,
                "min_p": 0.05,
            },
            thinking_options={
                "temperature": 0.6,
                "repeat_penalty": 1.0,
                "min_p": None,
            },
        )

        chunks = list(client.stream_chat([{"role": "user", "content": "hello"}], think_override=False))

        self.assertEqual(chunks, ["ok"])
        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["temperature"], 1.0)
        self.assertEqual(payload["options"]["repeat_penalty"], 1.15)
        self.assertEqual(payload["options"]["min_p"], 0.05)

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_retries_without_images_on_http_400(self, post_mock):
        error_response = requests.Response()
        error_response.status_code = 400
        error_response._content = b'{"error":"model does not support images"}'

        post_mock.side_effect = [
            _FakeStreamResponse(
                lines=[],
                http_error=requests.HTTPError(response=error_response),
            ),
            _FakeStreamResponse(
                lines=[
                    b'{"message":{"content":"ok"},"done":false}',
                    b'{"message":{},"done":true,"done_reason":"stop"}',
                ],
            ),
        ]

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
        )

        chunks = list(
            client.stream_chat(
                [{"role": "user", "content": "hello", "images": ["aGVsbG8="]}]
            )
        )

        self.assertEqual(chunks, ["ok"])
        self.assertEqual(post_mock.call_count, 2)
        self.assertEqual(post_mock.call_args.kwargs["json"]["messages"][0]["content"], "hello")
        self.assertTrue(client.last_stream_dropped_current_images)
        self.assertEqual(client.last_stream_dropped_current_images_count, 1)

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_retries_without_images_on_http_500(self, post_mock):
        error_response = requests.Response()
        error_response.status_code = 500
        error_response._content = b'{"error":"internal server error"}'

        post_mock.side_effect = [
            _FakeStreamResponse(
                lines=[],
                http_error=requests.HTTPError(response=error_response),
            ),
            _FakeStreamResponse(
                lines=[
                    b'{"message":{"content":"ok"},"done":false}',
                    b'{"message":{},"done":true,"done_reason":"stop"}',
                ],
            ),
        ]

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
        )

        chunks = list(
            client.stream_chat(
                [{"role": "user", "content": "hello", "images": ["aGVsbG8="]}]
            )
        )

        self.assertEqual(chunks, ["ok"])
        self.assertEqual(post_mock.call_count, 2)
        self.assertEqual(post_mock.call_args.kwargs["json"]["messages"][0]["content"], "hello")
        self.assertTrue(client.last_stream_dropped_current_images)

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_keeps_images_enabled_after_single_bad_image(self, post_mock):
        error_response = requests.Response()
        error_response.status_code = 400
        error_response._content = b'{"error":"invalid image data"}'

        post_mock.side_effect = [
            _FakeStreamResponse(
                lines=[],
                http_error=requests.HTTPError(response=error_response),
            ),
            _FakeStreamResponse(
                lines=[
                    b'{"message":{"content":"first"},"done":false}',
                    b'{"message":{},"done":true,"done_reason":"stop"}',
                ],
            ),
            _FakeStreamResponse(
                lines=[
                    b'{"message":{"content":"second"},"done":false}',
                    b'{"message":{},"done":true,"done_reason":"stop"}',
                ],
            ),
        ]

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
        )

        list(client.stream_chat([{"role": "user", "content": "hello", "images": ["aGVsbG8="]}]))
        list(client.stream_chat([{"role": "user", "content": "again", "images": ["d29ybGQ="]}]))

        self.assertEqual(post_mock.call_count, 3)
        last_messages = post_mock.call_args.kwargs["json"]["messages"]
        self.assertEqual(post_mock.call_args.args[0], "http://localhost:11434/api/chat")
        self.assertEqual(last_messages[0]["content"], "again")
        self.assertEqual(last_messages[0]["images"], ["d29ybGQ="])

    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_stream_disables_images_when_model_lacks_vision_support(self, post_mock):
        error_response = requests.Response()
        error_response.status_code = 400
        error_response._content = b'{"error":"model does not support images"}'

        post_mock.side_effect = [
            _FakeStreamResponse(
                lines=[],
                http_error=requests.HTTPError(response=error_response),
            ),
            _FakeStreamResponse(
                lines=[
                    b'{"message":{"content":"first"},"done":false}',
                    b'{"message":{},"done":true,"done_reason":"stop"}',
                ],
            ),
            _FakeStreamResponse(
                lines=[
                    _ndjson_line('{"message":{"content":"second"},"done":false}'),
                    _ndjson_line('{"message":{},"done":true,"done_reason":"stop"}'),
                ],
            ),
        ]

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
        )

        list(client.stream_chat([{"role": "user", "content": "hello", "images": ["aGVsbG8="]}]))
        self.assertTrue(client.last_stream_dropped_current_images)
        list(client.stream_chat([{"role": "user", "content": "again", "images": ["d29ybGQ="]}]))

        self.assertEqual(post_mock.call_count, 3)
        last_messages = post_mock.call_args.kwargs["json"]["messages"]
        self.assertEqual(last_messages[0]["content"], "again")

    @patch("app.tools.web_search.time.sleep", return_value=None)
    @patch("app.tools.web_search.requests.get")
    def test_web_search_retries_on_connection_error_then_succeeds(
        self, get_mock, _sleep_mock
    ):
        get_mock.side_effect = [
            requests.ConnectionError("conn"),
            _FakeResponse(
                data={
                    "results": [
                        {
                            "title": "T",
                            "url": "https://example.com",
                            "content": "C",
                        }
                    ]
                }
            ),
        ]

        client = SearXNGClient(
            base_url="http://localhost:8080",
            timeout=0.1,
            max_retries=2,
            retry_backoff_s=0.0,
        )
        results = client.search("query", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "T")
        self.assertEqual(get_mock.call_count, 2)

    @patch("app.tools.web_search.time.sleep", return_value=None)
    @patch("app.tools.web_search.requests.get")
    def test_web_search_does_not_retry_on_http_400(self, get_mock, _sleep_mock):
        response = requests.Response()
        response.status_code = 400
        get_mock.side_effect = requests.HTTPError(response=response)

        client = SearXNGClient(
            base_url="http://localhost:8080",
            timeout=0.1,
            max_retries=3,
            retry_backoff_s=0.0,
        )

        with self.assertRaises(requests.HTTPError):
            client.search("query", limit=5)

        self.assertEqual(get_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
