import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from app.telemetry import TelemetryRecorder


class TelemetryRecorderTests(unittest.TestCase):
    def test_none_does_not_create_file_or_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "telemetry.jsonl"
            recorder = TelemetryRecorder("none", file_path=path)

            self.assertIsNone(recorder.begin_turn("session"))
            recorder.record_model_call(prompt_tokens=10)
            recorder.close()

            self.assertFalse(path.exists())

    def test_minimal_logs_summary_without_creating_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "telemetry.jsonl"
            recorder = TelemetryRecorder("minimal", file_path=path)
            handle = recorder.begin_turn("session")
            recorder.record_model_call(
                session_id="session",
                status="success",
                model_name="test-model",
                prompt_tokens=10,
                completion_tokens=5,
                eval_duration_ms=250.0,
                total_model_duration_ms=400.0,
                time_to_first_token_ms=50.0,
            )

            with patch("app.telemetry.logger.info") as info:
                turn = recorder.end_turn(handle, status="success")
            recorder.close()

            self.assertEqual(turn["prompt_tokens"], 10)
            self.assertEqual(turn["completion_tokens"], 5)
            self.assertEqual(turn["decode_tokens_per_s"], 20.0)
            info.assert_called_once()
            self.assertFalse(path.exists())

    def test_full_writes_model_and_turn_jsonl_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "telemetry.jsonl"
            recorder = TelemetryRecorder("full", file_path=path)
            recorder.set_model_metadata(quantization="Q4_K_M")
            handle = recorder.begin_turn("session")
            recorder.record_retrieval_duration(12.5, session_id="session")
            recorder.record_tool_duration(7.25, session_id="session")
            recorder.record_model_call(
                session_id="session",
                status="success",
                model_name="test-model",
                prompt_tokens=10,
                completion_tokens=5,
                eval_duration_ms=250.0,
                total_model_duration_ms=400.0,
                context_tokens_total=15,
            )
            recorder.end_turn(handle, status="success")
            recorder.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([item["event"] for item in records], ["model_call", "turn_complete"])
            self.assertEqual(records[0]["quantization"], "Q4_K_M")
            self.assertEqual(records[0]["context_tokens_total"], 15)
            self.assertEqual(records[1]["retrieval_duration_ms"], 12.5)
            self.assertEqual(records[1]["tool_duration_ms"], 7.25)

    def test_session_correlation_survives_worker_thread_changes(self):
        recorder = TelemetryRecorder("minimal")
        handle = recorder.begin_turn("session")

        with ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(
                recorder.record_model_call,
                session_id="session",
                status="success",
                model_name="test-model",
                prompt_tokens=3,
                completion_tokens=2,
                eval_duration_ms=100.0,
                total_model_duration_ms=150.0,
            ).result()
            pool.submit(
                recorder.record_tool_duration,
                25.0,
                session_id="session",
            ).result()

        with patch("app.telemetry.logger.info"):
            turn = recorder.end_turn(handle, status="success")
        recorder.close()

        self.assertEqual(turn["model_call_count"], 1)
        self.assertEqual(turn["prompt_tokens"], 3)
        self.assertEqual(turn["tool_duration_ms"], 25.0)


if __name__ == "__main__":
    unittest.main()
