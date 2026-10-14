import copy
import json
import logging
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import requests

import app.logging as logging_module
from app.core.response_generator import ResponseGenerator, inject_hidden_system_message
from app.llm.ollama_stream import OllamaClient
from tests.support import make_response


class OllamaTraceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        logger = logging.getLogger("trace")
        self.stack.enter_context(patch.object(logger, "isEnabledFor", return_value=True))
        self.debug = self.stack.enter_context(patch.object(logger, "debug"))
        self.stack.enter_context(patch.object(logging_module, "_TRACE_MAX_STRING_CHARS", None))
        self.stack.enter_context(patch.object(logging_module, "_TRACE_MAX_EVENT_CHARS", None))

    def client(self, mode):
        message = {"content": "answer", "thinking": "fixture analysis"}
        if mode == "blocking":
            response = make_response(data={"message": message, "done": True})
        else:
            response = make_response(lines=[
                json.dumps({"message": {"thinking": "fixture "}, "done": False}).encode(),
                json.dumps({"message": {"thinking": "analysis"}, "done": False}).encode(),
                json.dumps({"message": {"content": "answer"}, "done": True}).encode(),
            ])
        client = OllamaClient(model="test", host="http://localhost:11434")
        self.addCleanup(client.close)
        client.session.post = Mock(return_value=response)
        return client

    def records(self):
        return [json.loads(call.args[0]) for call in self.debug.call_args_list]

    def test_final_payload_and_optional_thinking_in_all_modes(self):
        for mode in ("blocking", "buffered", "streaming"):
            for thinking_enabled in (False, True):
                with self.subTest(mode=mode, thinking_enabled=thinking_enabled), \
                        patch.object(logging_module, "_TRACE_INCLUDE_THINKING", thinking_enabled):
                    self.debug.reset_mock()
                    messages = [{"role": "system", "content": "fixture prompt " * 1500},
                                {"role": "user", "content": "inspect", "images": ["fixture-bytes"]}]
                    ResponseGenerator.__new__(ResponseGenerator)._inject_late_routing_system_message(messages)
                    inject_hidden_system_message(messages, "fixture belief catalog")
                    original = copy.deepcopy(messages)
                    tools = [{"type": "function", "function": {
                        "name": "fixture_tool", "parameters": {"type": "object", "properties": {}}}}]
                    client = self.client(mode)
                    kwargs = {"tools": tools, "telemetry_session_id": "fixture-session"}
                    if mode == "blocking":
                        result = client.chat(messages, **kwargs)
                        self.assertEqual(result["thinking"], "fixture analysis")
                    elif mode == "buffered":
                        result = client.chat_buffered(messages, **kwargs)
                        self.assertEqual(result["thinking"], "fixture analysis")
                    else:
                        self.assertIn("answer", list(client.stream_chat(messages, **kwargs)))
                    records = self.records()
                    request = next(r for r in records if r["event"] == "request_payload")
                    logged_payload = request["payload"]["request"]
                    sent_payload = client.session.post.call_args.kwargs["json"]
                    self.assertEqual(sent_payload["messages"], original)
                    self.assertEqual(messages, original)
                    self.assertEqual(logged_payload["messages"][:3], original[:3])
                    self.assertTrue(logged_payload["messages"][3]["images"][0].startswith("<omitted images"))
                    self.assertEqual(logged_payload["tools"], tools)
                    self.assertEqual(request["session_id"], "fixture-session")
                    inference_id = request["payload"]["inference_id"]
                    self.assertTrue(inference_id)
                    self.assertTrue(request["payload"]["request_id"])
                    self.assertTrue(all(r["payload"]["inference_id"] == inference_id for r in records))
                    thinking = [r for r in records if r["event"] == "thinking_output"]
                    self.assertEqual(len(thinking), int(thinking_enabled))
                    if thinking_enabled:
                        self.assertEqual(thinking[0]["payload"]["thinking"], "fixture analysis")
                    else:
                        self.assertNotIn("fixture analysis", json.dumps(records))

    def test_retries_record_each_actual_attempt_with_shared_inference_id(self):
        client = self.client("blocking")
        response = client.session.post.return_value
        client.session.post.side_effect = [requests.ConnectTimeout("fixture"), response]
        with patch("app.llm.ollama_stream.time.sleep"):
            client.chat([], max_retries_override=1, telemetry_session_id="fixture-session")
        records = self.records()
        requests_logged = [r["payload"] for r in records if r["event"] == "request_payload"]
        self.assertEqual([r["transport_attempt"] for r in requests_logged], [1, 2])
        self.assertEqual(len({r["request_id"] for r in requests_logged}), 2)
        self.assertEqual(len({r["inference_id"] for r in requests_logged}), 1)
        retry = next(r for r in records if r["event"] == "request_retry")
        self.assertEqual(retry["payload"]["inference_id"], requests_logged[0]["inference_id"])
        self.assertEqual(retry["session_id"], "fixture-session")

    def test_image_fallback_logs_each_changed_payload_without_rewriting_prior_trace(self):
        for mode in ("blocking", "streaming"):
            with self.subTest(mode=mode):
                self.debug.reset_mock()
                client = self.client(mode)
                success = client.session.post.return_value
                client.session.post.side_effect = [
                    make_response(data={"error": "model does not support images"}, status_code=400),
                    success,
                ]
                messages = [{"role": "user", "content": "inspect", "images": ["fixture-bytes"]}]
                if mode == "blocking":
                    client.chat(messages, telemetry_session_id="fixture-session")
                else:
                    list(client.stream_chat(messages, telemetry_session_id="fixture-session"))
                attempts = [r["payload"] for r in self.records() if r["event"] == "request_payload"]
                self.assertEqual(len(attempts), 2)
                self.assertIn("images", attempts[0]["request"]["messages"][0])
                self.assertNotIn("images", attempts[1]["request"]["messages"][0])
                self.assertEqual(attempts[0]["inference_id"], attempts[1]["inference_id"])
                self.assertNotEqual(attempts[0]["request_id"], attempts[1]["request_id"])
                self.assertEqual(messages[0]["images"], ["fixture-bytes"])


if __name__ == "__main__":
    unittest.main()
