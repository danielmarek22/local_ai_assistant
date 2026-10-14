import subprocess
import copy
import json
import logging
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app.logging as logging_module


class TraceEventTests(unittest.TestCase):
    def setUp(self):
        logging_module._WARNED_UNREGISTERED_TRACE_EVENTS.clear()

    def test_unregistered_trace_event_logs_warning_once(self):
        trace_logger = Mock()
        trace_logger.isEnabledFor.return_value = True

        module_logger = Mock()

        def get_logger(name=None):
            if name == logging_module._TRACE_LOGGER_NAME:
                return trace_logger
            return module_logger

        with patch("app.logging.logging.getLogger", side_effect=get_logger):
            logging_module.trace_event("typo_component", "typo_event", payload={"x": 1})
            logging_module.trace_event("typo_component", "typo_event", payload={"x": 2})

        module_logger.warning.assert_called_once_with(
            "Unregistered trace event convention: %s",
            "typo_component.typo_event",
        )
        self.assertEqual(trace_logger.debug.call_count, 2)

    def test_buffered_requests_use_registered_trace_conventions(self):
        trace_logger = Mock()
        trace_logger.isEnabledFor.return_value = True
        module_logger = Mock()

        def get_logger(name=None):
            return trace_logger if name == logging_module._TRACE_LOGGER_NAME else module_logger

        with patch("app.logging.logging.getLogger", side_effect=get_logger):
            logging_module.trace_event("llm", "chat_request", payload={"buffered": True})
            logging_module.trace_event("llm", "chat_response", payload={"buffered": True})
            logging_module.trace_event(
                "orchestrator", "late_routing_inference_failure", payload={"phase": "initial"}
            )
            logging_module.trace_event("llm", "request_failure", payload={"retry": False})

        module_logger.warning.assert_not_called()
        self.assertEqual(trace_logger.debug.call_count, 4)

    def test_trace_preserves_long_prompt_and_omits_media_without_mutating_payload(self):
        payload = {"messages": [{"role": "system", "content": "prompt " * 3000},
                                {"role": "user", "content": "Inspect", "images": ["fixture-bytes"]}]}
        original = copy.deepcopy(payload)
        logger = logging.getLogger("trace")
        with patch.object(logger, "isEnabledFor", return_value=True), patch.object(logger, "debug") as debug, \
                patch.object(logging_module, "_TRACE_MAX_STRING_CHARS", None), \
                patch.object(logging_module, "_TRACE_MAX_EVENT_CHARS", None):
            logging_module.trace_event("llm", "request_payload", payload=payload)
        record = json.loads(debug.call_args.args[0])
        self.assertEqual(record["payload"]["messages"][0]["content"], payload["messages"][0]["content"])
        self.assertTrue(record["payload"]["messages"][1]["images"][0].startswith("<omitted images"))
        self.assertEqual(payload, original)

    def test_explicit_trace_caps_keep_valid_json_and_identify_log_only_truncation(self):
        value = {"component": "llm", "event": "request_payload", "event_key": "llm.request_payload",
                 "session_id": "fixture-session", "payload": {"content": "quotes: \" \\ ś " * 2000}}
        capped = logging_module.serialize_for_debug(value, max_chars=1024, max_string_chars=None)
        record = json.loads(capped)
        self.assertLessEqual(len(capped), 1024)
        self.assertEqual(record["event_key"], "llm.request_payload")
        self.assertEqual(record["session_id"], "fixture-session")
        self.assertTrue(record["log_truncation"]["log_only"])
        self.assertEqual(record["log_truncation"]["reason"], "event_limit")
        string_capped = json.loads(logging_module.serialize_for_debug(value, max_chars=None, max_string_chars=20))
        self.assertIn("log-only truncation", string_capped["payload"]["content"])

    def test_thinking_is_optional_and_logged_as_its_own_event(self):
        logger = logging.getLogger("trace")
        with patch.object(logger, "isEnabledFor", return_value=True), patch.object(logger, "debug") as debug:
            with patch.object(logging_module, "_TRACE_INCLUDE_THINKING", False):
                logging_module.trace_thinking_output("fixture analysis", inference_id="fixture-inference")
                debug.assert_not_called()
            with patch.object(logging_module, "_TRACE_INCLUDE_THINKING", True):
                logging_module.trace_thinking_output("fixture analysis", session_id="fixture-session",
                                                    inference_id="fixture-inference")
        record = json.loads(debug.call_args.args[0])
        self.assertEqual(record["event_key"], "llm.thinking_output")
        self.assertEqual(record["session_id"], "fixture-session")
        self.assertEqual(record["payload"]["inference_id"], "fixture-inference")
        self.assertEqual(record["payload"]["thinking"], "fixture analysis")


class LoggingLifecycleTests(unittest.TestCase):
    def test_logging_can_be_reset_and_reconfigured_without_leaking_handlers(self):
        repository_root = Path(__file__).resolve().parents[1]
        script = textwrap.dedent(
            """
            import logging
            import sys
            from pathlib import Path

            import app.logging as astra_logging

            first_dir = Path(sys.argv[1])
            second_dir = Path(sys.argv[2])
            root = logging.getLogger()
            trace = logging.getLogger("trace")
            initial_root_level = root.level
            initial_trace_state = (trace.level, trace.propagate, trace.disabled)

            astra_logging.setup_logging(
                log_dir=str(first_dir),
                console_level=logging.CRITICAL,
                trace_max_string_chars=9000,
                trace_max_event_chars=50000,
                trace_include_thinking=True,
            )
            owned_handlers = tuple(astra_logging._CONFIGURED_HANDLERS)
            assert astra_logging._LOGGING_CONFIGURED
            assert astra_logging._TRACE_MAX_STRING_CHARS == 9000
            assert astra_logging._TRACE_MAX_EVENT_CHARS == 50000
            assert astra_logging._TRACE_INCLUDE_THINKING
            assert len(owned_handlers) == 3
            assert (first_dir / "assistant.log").is_file()
            assert (first_dir / "trace.log").is_file()
            foreign_handler = logging.NullHandler()
            root.addHandler(foreign_handler)

            astra_logging.setup_logging(
                log_dir=str(second_dir),
                console_level=logging.CRITICAL,
            )
            assert not second_dir.exists()

            astra_logging.reset_logging()
            assert not astra_logging._LOGGING_CONFIGURED
            assert astra_logging._CONFIGURED_HANDLERS == ()
            assert astra_logging._TRACE_MAX_STRING_CHARS is None
            assert astra_logging._TRACE_MAX_EVENT_CHARS is None
            assert not astra_logging._TRACE_INCLUDE_THINKING
            assert all(handler not in root.handlers for handler in owned_handlers)
            assert all(handler not in trace.handlers for handler in owned_handlers)
            assert foreign_handler in root.handlers
            assert root.level == initial_root_level
            assert (trace.level, trace.propagate, trace.disabled) == initial_trace_state
            root.removeHandler(foreign_handler)
            foreign_handler.close()

            astra_logging.reset_logging()
            astra_logging.setup_logging(
                log_dir=str(second_dir),
                console_level=logging.CRITICAL,
                trace_enabled=False,
            )
            assert (second_dir / "assistant.log").is_file()
            assert not (second_dir / "trace.log").exists()
            astra_logging.reset_logging()
            """
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(Path(temp_dir) / "first"),
                    str(Path(temp_dir) / "second"),
                ],
                cwd=repository_root,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
