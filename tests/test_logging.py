import unittest
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


if __name__ == "__main__":
    unittest.main()
