import logging
import logging.config
import json
import sys
from pathlib import Path
from typing import Any


_LOGGING_CONFIGURED = False
_TRACE_LOGGER_NAME = "trace"
_SENSITIVE_BINARY_KEYS = {"base64_data", "data", "images"}
_DEFAULT_MAX_STRING_CHARS = 4000
_DEFAULT_MAX_LOG_CHARS = 12000

# Trace naming convention table.
# Canonical grep token format in logs/trace.log:
#   "event_key": "<component>.<event>"
TRACE_EVENT_CONVENTIONS: dict[str, tuple[str, ...]] = {
    "orchestrator": (
        "turn_input",
        "memory_retrieval",
        "plan_result",
        "assistant_response",
        "planner_input",
        "llm_stream_complete",
    ),
    "context_builder": ("context_built",),
    "llm_planner": (
        "planner_call",
        "planner_timeout",
        "planner_result",
        "planner_parse_failure",
    ),
    "llm": (
        "chat_request",
        "chat_response",
        "stream_request",
        "stream_response",
    ),
    "web_search": (
        "probe_start",
        "probe_result",
        "search_request",
        "search_results",
        "tool_run",
        "tool_summary",
    ),
    "tool_executor": (
        "tool_call",
        "tool_result",
    ),
    "memory_retriever": ("retrieval_result",),
    "memory_action": (
        "memory_action_skipped",
        "memory_action_applied",
    ),
    "turn_finalizer": (
        "summarization_check",
        "summary_input",
        "summary_saved",
    ),
    "history_summarizer": ("summary_generated",),
    "search_summarizer": ("summary_generated",),
    "image_summarizer": ("summary_generated",),
    "memory_store": (
        "memory_saved",
        "semantic_query_result",
    ),
    "chat_history": (
        "history_add",
        "attachment_stored",
        "attachment_summary",
        "episodic_search",
        "recent_history",
    ),
    "summary_store": (
        "summary_get",
        "summary_set",
    ),
}


def _coerce_level(level: int | str | None, default: int) -> int:
    if level is None:
        return default
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = logging.getLevelName(level.upper())
        if isinstance(normalized, int):
            return normalized
    return default


def _sanitize_for_debug(
    value: Any,
    *,
    parent_key: str | None = None,
    max_string_chars: int = _DEFAULT_MAX_STRING_CHARS,
    max_depth: int = 8,
    _depth: int = 0,
) -> Any:
    if _depth >= max_depth:
        return "<max-depth-reached>"

    if isinstance(value, dict):
        return {
            str(key): _sanitize_for_debug(
                item,
                parent_key=str(key),
                max_string_chars=max_string_chars,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _sanitize_for_debug(
                item,
                parent_key=parent_key,
                max_string_chars=max_string_chars,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if isinstance(value, str):
        if parent_key in _SENSITIVE_BINARY_KEYS:
            return f"<omitted {parent_key} len={len(value)}>"
        if len(value) > max_string_chars:
            return f"{value[:max_string_chars]}... [truncated {len(value) - max_string_chars} chars]"
        return value

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if hasattr(value, "model_dump"):
        return _sanitize_for_debug(
            value.model_dump(),
            parent_key=parent_key,
            max_string_chars=max_string_chars,
            max_depth=max_depth,
            _depth=_depth + 1,
        )

    if hasattr(value, "__dict__"):
        return _sanitize_for_debug(
            vars(value),
            parent_key=parent_key,
            max_string_chars=max_string_chars,
            max_depth=max_depth,
            _depth=_depth + 1,
        )

    return repr(value)


def serialize_for_debug(
    value: Any,
    *,
    max_chars: int = _DEFAULT_MAX_LOG_CHARS,
    max_string_chars: int = _DEFAULT_MAX_STRING_CHARS,
) -> str:
    sanitized = _sanitize_for_debug(value, max_string_chars=max_string_chars)
    try:
        rendered = json.dumps(sanitized, ensure_ascii=True, indent=2, sort_keys=True)
    except TypeError:
        rendered = repr(sanitized)

    if len(rendered) > max_chars:
        return f"{rendered[:max_chars]}... [truncated {len(rendered) - max_chars} chars]"
    return rendered


def setup_logging(
    level=logging.INFO,
    log_dir: str = "logs",
    *,
    console_level: int | str | None = None,
    file_level: int | str | None = logging.INFO,
    file_name: str = "assistant.log",
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    trace_enabled: bool = True,
    trace_level: int | str | None = logging.DEBUG,
    trace_file_name: str = "trace.log",
    trace_max_bytes: int | None = None,
    trace_backup_count: int | None = None,
):
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        logging.getLogger(__name__).debug("Logging already configured, skipping")
        return

    resolved_console_level = _coerce_level(console_level, _coerce_level(level, logging.INFO))
    resolved_file_level = _coerce_level(file_level, logging.INFO)
    resolved_trace_level = _coerce_level(trace_level, logging.DEBUG)
    root_level = min(resolved_console_level, resolved_file_level)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / file_name
    trace_file = log_path / trace_file_name

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "default",
            "level": resolved_console_level,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
            "formatter": "default",
            "level": resolved_file_level,
        },
    }

    loggers = {}
    if trace_enabled:
        handlers["trace_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(trace_file),
            "maxBytes": trace_max_bytes or max_bytes,
            "backupCount": trace_backup_count or backup_count,
            "encoding": "utf-8",
            "formatter": "default",
            "level": resolved_trace_level,
        }
        loggers[_TRACE_LOGGER_NAME] = {
            "level": resolved_trace_level,
            "handlers": ["trace_file"],
            "propagate": False,
        }
    else:
        loggers[_TRACE_LOGGER_NAME] = {
            "level": logging.CRITICAL,
            "handlers": [],
            "propagate": False,
        }

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": handlers,
        "loggers": loggers,
        "root": {
            "level": root_level,
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(logging_config)

    root = logging.getLogger()
    root.info("Logging initialized")
    root.info(
        "Log levels: console=%s file=%s",
        logging.getLevelName(resolved_console_level),
        logging.getLevelName(resolved_file_level),
    )
    root.info("Log file: %s", log_file.resolve())
    if trace_enabled:
        root.info(
            "Trace log: %s (level=%s)",
            trace_file.resolve(),
            logging.getLevelName(resolved_trace_level),
        )

    _LOGGING_CONFIGURED = True


def setup_logging_from_config(config: dict | None = None) -> None:
    config = config or {}
    setup_logging(
        level=config.get("level", logging.INFO),
        console_level=config.get("console_level"),
        file_level=config.get("file_level", logging.INFO),
        log_dir=config.get("dir", "logs"),
        file_name=config.get("file_name", "assistant.log"),
        max_bytes=int(config.get("max_bytes", 10_000_000)),
        backup_count=int(config.get("backup_count", 5)),
        trace_enabled=bool(config.get("trace_enabled", True)),
        trace_level=config.get("trace_level", logging.DEBUG),
        trace_file_name=config.get("trace_file_name", "trace.log"),
        trace_max_bytes=(
            int(config["trace_max_bytes"]) if config.get("trace_max_bytes") is not None else None
        ),
        trace_backup_count=(
            int(config["trace_backup_count"]) if config.get("trace_backup_count") is not None else None
        ),
    )


def trace_event(
    component: str,
    event: str,
    *,
    session_id: str | None = None,
    payload: Any = None,
    **fields: Any,
) -> None:
    trace_logger = logging.getLogger(_TRACE_LOGGER_NAME)
    if not trace_logger.isEnabledFor(logging.DEBUG):
        return

    known_component = component in TRACE_EVENT_CONVENTIONS
    known_event = event in TRACE_EVENT_CONVENTIONS.get(component, ())

    record: dict[str, Any] = {
        "component": component,
        "event": event,
        "event_key": f"{component}.{event}",
    }
    if not known_component or not known_event:
        record["convention"] = "unregistered"
    if session_id is not None:
        record["session_id"] = session_id
    if fields:
        record["fields"] = fields
    if payload is not None:
        record["payload"] = payload

    trace_logger.debug(serialize_for_debug(record))
