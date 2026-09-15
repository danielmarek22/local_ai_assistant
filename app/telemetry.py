from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal


TelemetryMode = Literal["none", "minimal", "full"]

logger = logging.getLogger("model_telemetry")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _rounded(value: float | int | None) -> float | int | None:
    if isinstance(value, float):
        return round(value, 2)
    return value


@dataclass
class _TurnCapture:
    turn_id: str
    session_id: str
    turn_kind: str
    started: float
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieval_duration_ms: float = 0.0
    tool_duration_ms: float = 0.0


@dataclass(frozen=True)
class TurnHandle:
    capture: _TurnCapture


class TelemetryRecorder:
    """Collect inexpensive turn metrics and optionally persist full JSONL events."""

    def __init__(
        self,
        mode: TelemetryMode = "none",
        *,
        file_path: str | Path = "logs/model-telemetry.jsonl",
        max_bytes: int = 10_000_000,
        backup_count: int = 5,
    ) -> None:
        self.mode: TelemetryMode = mode
        self._handler: RotatingFileHandler | None = None
        self._write_lock = threading.Lock()
        self._turn_lock = threading.Lock()
        self._turns_by_session: dict[str, _TurnCapture] = {}
        self._model_metadata: dict[str, Any] = {"quantization": None}

        if mode == "full":
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._handler = handler

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def full(self) -> bool:
        return self.mode == "full"

    def set_model_metadata(self, **metadata: Any) -> None:
        if self.full:
            self._model_metadata.update(
                {key: value for key, value in metadata.items() if value is not None}
            )

    def begin_turn(self, session_id: str, *, turn_kind: str = "user") -> TurnHandle | None:
        if not self.enabled:
            return None
        capture = _TurnCapture(
            turn_id=uuid.uuid4().hex,
            session_id=session_id,
            turn_kind=turn_kind,
            started=time.perf_counter(),
        )
        with self._turn_lock:
            self._turns_by_session[session_id] = capture
        return TurnHandle(capture=capture)

    def record_retrieval_duration(self, duration_ms: float, *, session_id: str) -> None:
        capture = self._capture_for_session(session_id)
        if capture is not None:
            capture.retrieval_duration_ms += max(0.0, duration_ms)

    def record_tool_duration(self, duration_ms: float, *, session_id: str) -> None:
        capture = self._capture_for_session(session_id)
        if capture is not None:
            capture.tool_duration_ms += max(0.0, duration_ms)

    def record_model_call(self, *, session_id: str | None = None, **metrics: Any) -> None:
        if not self.enabled:
            return
        capture = self._capture_for_session(session_id) if session_id else None
        record = {
            "schema_version": 1,
            "event": "model_call",
            "timestamp": _utc_timestamp(),
            "turn_id": capture.turn_id if capture is not None else None,
            "session_id": capture.session_id if capture is not None else None,
            "call_id": uuid.uuid4().hex,
            **self._model_metadata,
            **{key: _rounded(value) for key, value in metrics.items()},
        }
        if capture is not None:
            capture.model_calls.append(record)
        if self.full:
            self._write(record)

    def end_turn(
        self,
        handle: TurnHandle | None,
        *,
        status: str,
        error_category: str | None = None,
    ) -> dict[str, Any] | None:
        if handle is None:
            return None
        capture = handle.capture
        try:
            prompt_tokens = sum(int(call.get("prompt_tokens") or 0) for call in capture.model_calls)
            completion_tokens = sum(
                int(call.get("completion_tokens") or 0) for call in capture.model_calls
            )
            eval_duration_ms = sum(
                float(call.get("eval_duration_ms") or 0.0) for call in capture.model_calls
            )
            total_model_duration_ms = sum(
                float(call.get("total_model_duration_ms") or 0.0)
                for call in capture.model_calls
            )
            successful_calls = [
                call for call in capture.model_calls if call.get("status") == "success"
            ]
            final_call = successful_calls[-1] if successful_calls else None
            decode_tokens_per_s = (
                completion_tokens / (eval_duration_ms / 1000.0)
                if completion_tokens and eval_duration_ms > 0
                else None
            )
            record = {
                "schema_version": 1,
                "event": "turn_complete",
                "timestamp": _utc_timestamp(),
                "turn_id": capture.turn_id,
                "session_id": capture.session_id,
                "turn_kind": capture.turn_kind,
                "status": status,
                "error_category": error_category,
                "model_call_count": len(capture.model_calls),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "time_to_first_token_ms": (
                    final_call.get("time_to_first_token_ms") if final_call else None
                ),
                "decode_tokens_per_s": _rounded(decode_tokens_per_s),
                "total_model_duration_ms": _rounded(total_model_duration_ms),
                "total_turn_duration_ms": _rounded(
                    (time.perf_counter() - capture.started) * 1000
                ),
                "retrieval_duration_ms": _rounded(capture.retrieval_duration_ms),
                "tool_duration_ms": _rounded(capture.tool_duration_ms),
                "model_name": final_call.get("model_name") if final_call else None,
            }
            self._log_summary(record)
            if self.full:
                self._write(record)
            return record
        finally:
            with self._turn_lock:
                if self._turns_by_session.get(capture.session_id) is capture:
                    self._turns_by_session.pop(capture.session_id, None)

    def close(self) -> None:
        with self._write_lock:
            if self._handler is not None:
                self._handler.close()
                self._handler = None

    def _log_summary(self, record: dict[str, Any]) -> None:
        ttft = record.get("time_to_first_token_ms")
        decode = record.get("decode_tokens_per_s")
        logger.info(
            "Model: %s | calls=%d | tokens=%s\u2192%s | TTFT=%s | decode=%s | model=%.2f s | turn=%.2f s",
            record.get("model_name") or "unknown",
            record["model_call_count"],
            f"{record['prompt_tokens']:,}",
            f"{record['completion_tokens']:,}",
            f"{ttft:.0f} ms" if isinstance(ttft, (int, float)) else "n/a",
            f"{decode:.1f} tok/s" if isinstance(decode, (int, float)) else "n/a",
            float(record["total_model_duration_ms"]) / 1000.0,
            float(record["total_turn_duration_ms"]) / 1000.0,
        )

    def _write(self, record: dict[str, Any]) -> None:
        rendered = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        with self._write_lock:
            if self._handler is not None:
                log_record = logging.LogRecord(
                    name="model_telemetry",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=0,
                    msg=rendered,
                    args=(),
                    exc_info=None,
                )
                self._handler.emit(log_record)

    def _capture_for_session(self, session_id: str) -> _TurnCapture | None:
        with self._turn_lock:
            return self._turns_by_session.get(session_id)
