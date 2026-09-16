from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal


TelemetryMode = Literal["none", "minimal", "full", "extended"]

logger = logging.getLogger("model_telemetry")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _rounded(value: float | int | None) -> float | int | None:
    if isinstance(value, float):
        return round(value, 2)
    return value


def _read_kib_value(fields: dict[str, str], key: str) -> int | None:
    raw = fields.get(key)
    if raw is None:
        return None
    try:
        return int(raw.split()[0]) * 1024
    except (TypeError, ValueError, IndexError):
        return None


def _optional_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


class _ExtendedSystemProbe:
    """Best-effort post-inference host and NVIDIA GPU snapshot."""

    _GPU_FIELDS = {
        "gpu_count": 0,
        "gpu_vram_used_bytes": None,
        "gpu_vram_free_bytes": None,
        "gpu_utilization_percent": None,
        "gpu_clock_mhz": None,
        "gpu_temperature_c": None,
        "gpu_power_w": None,
        "gpu_performance_state": None,
        "gpus": [],
    }

    def __init__(self) -> None:
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._gpu_unavailable = self._nvidia_smi is None
        self._gpu_lock = threading.Lock()

    def collect(self) -> dict[str, Any]:
        return {
            "system_sample_timing": "post_model_call",
            **self._host_metrics(),
            **self._gpu_metrics(),
        }

    @staticmethod
    def _host_metrics() -> dict[str, int | None]:
        result: dict[str, int | None] = {
            "process_rss_bytes": None,
            "system_ram_used_bytes": None,
            "system_ram_available_bytes": None,
            "swap_used_bytes": None,
        }
        try:
            status = {}
            for line in Path("/proc/self/status").read_text().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    status[key] = value.strip()
            result["process_rss_bytes"] = _read_kib_value(status, "VmRSS")
        except OSError:
            pass

        try:
            meminfo = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meminfo[key] = value.strip()
            total = _read_kib_value(meminfo, "MemTotal")
            available = _read_kib_value(meminfo, "MemAvailable")
            swap_total = _read_kib_value(meminfo, "SwapTotal")
            swap_free = _read_kib_value(meminfo, "SwapFree")
            result["system_ram_available_bytes"] = available
            if total is not None and available is not None:
                result["system_ram_used_bytes"] = max(0, total - available)
            if swap_total is not None and swap_free is not None:
                result["swap_used_bytes"] = max(0, swap_total - swap_free)
        except OSError:
            pass
        return result

    def _gpu_metrics(self) -> dict[str, Any]:
        with self._gpu_lock:
            if self._gpu_unavailable or self._nvidia_smi is None:
                return dict(self._GPU_FIELDS)
            try:
                completed = subprocess.run(
                    [
                        self._nvidia_smi,
                        "--query-gpu=index,name,memory.used,memory.free,utilization.gpu,"
                        "clocks.current.graphics,temperature.gpu,power.draw,pstate",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
            except (OSError, subprocess.SubprocessError):
                self._gpu_unavailable = True
                return dict(self._GPU_FIELDS)

        gpus = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 9:
                continue
            used_mib = _optional_float(parts[2])
            free_mib = _optional_float(parts[3])
            gpus.append(
                {
                    "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                    "name": parts[1],
                    "vram_used_bytes": int(used_mib * 1024 * 1024) if used_mib is not None else None,
                    "vram_free_bytes": int(free_mib * 1024 * 1024) if free_mib is not None else None,
                    "utilization_percent": _optional_float(parts[4]),
                    "clock_mhz": _optional_float(parts[5]),
                    "temperature_c": _optional_float(parts[6]),
                    "power_w": _optional_float(parts[7]),
                    "performance_state": parts[8] or None,
                }
            )
        if not gpus:
            return dict(self._GPU_FIELDS)

        def values(key: str) -> list[float]:
            return [float(item[key]) for item in gpus if item.get(key) is not None]

        used = values("vram_used_bytes")
        free = values("vram_free_bytes")
        utilization = values("utilization_percent")
        clocks = values("clock_mhz")
        temperatures = values("temperature_c")
        power = values("power_w")
        states = [str(item["performance_state"]) for item in gpus if item.get("performance_state")]
        return {
            "gpu_count": len(gpus),
            "gpu_vram_used_bytes": int(sum(used)) if used else None,
            "gpu_vram_free_bytes": int(sum(free)) if free else None,
            "gpu_utilization_percent": max(utilization) if utilization else None,
            "gpu_clock_mhz": max(clocks) if clocks else None,
            "gpu_temperature_c": max(temperatures) if temperatures else None,
            "gpu_power_w": sum(power) if power else None,
            "gpu_performance_state": ",".join(states) if states else None,
            "gpus": gpus,
        }


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
        self._extended_probe = _ExtendedSystemProbe() if mode == "extended" else None

        if mode in {"full", "extended"}:
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
        return self.mode in {"full", "extended"}

    @property
    def extended(self) -> bool:
        return self.mode == "extended"

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
        extended_metrics = self._extended_probe.collect() if self._extended_probe else {}
        record = {
            "schema_version": 1,
            "event": "model_call",
            "timestamp": _utc_timestamp(),
            "turn_id": capture.turn_id if capture is not None else None,
            "session_id": capture.session_id if capture is not None else None,
            "call_id": uuid.uuid4().hex,
            **self._model_metadata,
            **extended_metrics,
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
