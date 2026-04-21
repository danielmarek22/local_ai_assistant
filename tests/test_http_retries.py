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


class HttpRetryTests(unittest.TestCase):
    @patch("app.llm.ollama_stream.time.sleep", return_value=None)
    @patch("app.llm.ollama_stream.requests.Session.post")
    def test_ollama_chat_retries_on_timeout_then_succeeds(self, post_mock, _sleep_mock):
        post_mock.side_effect = [
            requests.Timeout("timeout"),
            _FakeResponse(
                data={"message": {"content": "ok"}},
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
        self.assertFalse(post_mock.call_args.kwargs["json"]["think"])

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
    def test_ollama_stream_uses_configured_thinking_level(self, post_mock):
        post_mock.return_value = _FakeResponse(
            data={"message": {"content": "ok"}},
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
                (
                    b'{"message":{"thinking":"Thinking Process:\\n\\n1. Hello"},"done":false}'
                ),
                b'{"message":{},"done":true}',
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
