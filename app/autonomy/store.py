from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.integrations import EventId, EventSpec, IntegrationEvent, ReplayPolicy
from app.paths import DATA_DIR, resolve_app_path


@dataclass(frozen=True)
class EventRecord:
    event: IntegrationEvent
    status: str
    priority: int
    attempts: int
    outcome_summary: str | None = None
    notification: dict[str, object] | None = None
    error: str | None = None


@dataclass(frozen=True)
class OperationRecord:
    invocation_id: str
    capability: str
    session_id: str
    status: str
    event_id: str | None
    root_event_id: str | None
    causation_id: str | None


class AutonomyStore:
    """Thread-safe durable journal for integration events and tool operations."""

    def __init__(self, path: str = str(DATA_DIR / "assistant.db")):
        self.path = path if path == ":memory:" else str(resolve_app_path(path))
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def append_event(self, event: IntegrationEvent, spec: EventSpec) -> str:
        now = datetime.now(timezone.utc)
        occurred_at = self._iso(event.occurred_at)
        root_event_id = event.root_event_id or event.event_id
        with self._lock:
            if event.deduplication_key and spec.coalesce_window_s > 0:
                cutoff = self._iso(now - timedelta(seconds=spec.coalesce_window_s))
                row = self._conn.execute(
                    """
                    SELECT event_id FROM integration_events
                    WHERE event_type = ? AND session_id IS ? AND deduplication_key = ?
                      AND status = 'pending' AND occurred_at >= ?
                    ORDER BY occurred_at DESC LIMIT 1
                    """,
                    (str(event.event), event.session_id, event.deduplication_key, cutoff),
                ).fetchone()
                if row is not None:
                    self._conn.execute(
                        """
                        UPDATE integration_events
                        SET payload_json = ?, attachments_json = ?, occurred_at = ?, updated_at = ?
                        WHERE event_id = ?
                        """,
                        (
                            self._json(dict(event.payload)),
                            self._json([asdict(item) for item in event.attachments]),
                            occurred_at,
                            self._iso(now),
                            row["event_id"],
                        ),
                    )
                    self._conn.commit()
                    return str(row["event_id"])

            self._conn.execute(
                """
                INSERT INTO integration_events (
                    event_id, event_type, source, session_id, payload_json, attachments_json,
                    status, priority, occurred_at, correlation_id, causation_id, root_event_id,
                    deduplication_key, replay_policy, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    str(event.event),
                    event.event.integration,
                    event.session_id,
                    self._json(dict(event.payload)),
                    self._json([asdict(item) for item in event.attachments]),
                    spec.priority,
                    occurred_at,
                    event.correlation_id,
                    event.causation_id,
                    root_event_id,
                    event.deduplication_key,
                    spec.replay_policy.value,
                    self._iso(now),
                    self._iso(now),
                ),
            )
            self._conn.commit()
        return event.event_id

    def get_event(self, event_id: str) -> EventRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM integration_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._event_record(row) if row else None

    def pending_event_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_id FROM integration_events
                WHERE status = 'pending' AND session_id IS NOT NULL
                ORDER BY priority ASC, occurred_at ASC, event_id ASC
                """
            ).fetchall()
        return [str(row["event_id"]) for row in rows]

    def claim_event(self, event_id: str) -> EventRecord | None:
        now = self._iso(datetime.now(timezone.utc))
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE integration_events
                SET status = 'processing', attempts = attempts + 1,
                    processing_started_at = ?, updated_at = ?
                WHERE event_id = ? AND status = 'pending'
                """,
                (now, now, event_id),
            )
            self._conn.commit()
            if cursor.rowcount != 1:
                return None
        return self.get_event(event_id)

    def complete_event(
        self,
        event_id: str,
        outcome_summary: str,
        notification: dict[str, object] | None = None,
    ) -> None:
        self._finish_event(event_id, "completed", outcome_summary, notification, None)

    def fail_event(
        self,
        event_id: str,
        error: str,
        status: str = "failed",
        notification: dict[str, object] | None = None,
    ) -> None:
        self._finish_event(event_id, status, None, notification, error)

    def recover_interrupted(self) -> tuple[int, int]:
        now = self._iso(datetime.now(timezone.utc))
        with self._lock:
            replayed = self._conn.execute(
                """
                UPDATE integration_events SET status = 'pending', processing_started_at = NULL,
                    updated_at = ?, error = 'Recovered after assistant restart'
                WHERE status = 'processing' AND replay_policy = ?
                """,
                (now, ReplayPolicy.SAFE.value),
            ).rowcount
            failed = self._conn.execute(
                """
                UPDATE integration_events SET status = 'failed', completed_at = ?, updated_at = ?,
                    error = 'Autonomous turn interrupted by assistant restart'
                WHERE status = 'processing' AND replay_policy = ?
                """,
                (now, now, ReplayPolicy.NEVER.value),
            ).rowcount
            self._conn.commit()
        return replayed, failed

    def chain_stats(self, root_event_id: str) -> tuple[int, datetime | None]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS count, MIN(occurred_at) AS started_at
                FROM integration_events WHERE root_event_id = ?
                """,
                (root_event_id,),
            ).fetchone()
        started = datetime.fromisoformat(row["started_at"]) if row and row["started_at"] else None
        return (int(row["count"]), started) if row else (0, None)

    def recent_context(self, session_id: str, max_chars: int) -> str | None:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_type, status, outcome_summary, error, completed_at
                FROM integration_events
                WHERE session_id = ? AND status IN ('completed', 'failed', 'discarded')
                ORDER BY COALESCE(completed_at, updated_at) DESC LIMIT 20
                """,
                (session_id,),
            ).fetchall()
        lines = []
        for row in reversed(rows):
            detail = row["outcome_summary"] or row["error"] or "No details"
            lines.append(f"{row['event_type']} [{row['status']}]: {detail}")
        content = "\n".join(lines)
        return content[:max(0, int(max_chars))] or None

    def begin_operation(
        self,
        invocation_id: str,
        capability: str,
        session_id: str,
        event_id: str | None,
        root_event_id: str | None,
        causation_id: str | None,
    ) -> None:
        now = self._iso(datetime.now(timezone.utc))
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO integration_operations (
                    invocation_id, capability, session_id, status, event_id,
                    root_event_id, causation_id, created_at, updated_at
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)
                """,
                (
                    invocation_id, capability, session_id, event_id,
                    root_event_id, causation_id, now, now,
                ),
            )
            self._conn.commit()

    def finish_operation(self, invocation_id: str, status: str, result: str | None = None) -> None:
        with self._lock:
            if status == "pending":
                self._conn.execute(
                    """
                    UPDATE integration_operations SET status = ?, result = ?, updated_at = ?
                    WHERE invocation_id = ? AND status = 'running'
                    """,
                    (status, result, self._iso(datetime.now(timezone.utc)), invocation_id),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE integration_operations SET status = ?, result = ?, updated_at = ?
                    WHERE invocation_id = ?
                    """,
                    (status, result, self._iso(datetime.now(timezone.utc)), invocation_id),
                )
            self._conn.commit()

    def get_operation(self, invocation_id: str) -> OperationRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM integration_operations WHERE invocation_id = ?",
                (invocation_id,),
            ).fetchone()
        if row is None:
            return None
        return OperationRecord(
            invocation_id=row["invocation_id"], capability=row["capability"],
            session_id=row["session_id"], status=row["status"], event_id=row["event_id"],
            root_event_id=row["root_event_id"], causation_id=row["causation_id"],
        )

    def pending_operations(self, capability_prefix: str | None = None) -> list[OperationRecord]:
        query = "SELECT * FROM integration_operations WHERE status IN ('running', 'pending')"
        parameters: tuple[object, ...] = ()
        if capability_prefix:
            query += " AND capability LIKE ?"
            parameters = (f"{capability_prefix}%",)
        query += " ORDER BY created_at, invocation_id"
        with self._lock:
            rows = self._conn.execute(query, parameters).fetchall()
        return [
            OperationRecord(
                invocation_id=row["invocation_id"], capability=row["capability"],
                session_id=row["session_id"], status=row["status"], event_id=row["event_id"],
                root_event_id=row["root_event_id"], causation_id=row["causation_id"],
            )
            for row in rows
        ]

    def has_event_deduplication_key(self, event_type: str, deduplication_key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1 FROM integration_events
                WHERE event_type = ? AND deduplication_key = ?
                LIMIT 1
                """,
                (event_type, deduplication_key),
            ).fetchone()
        return row is not None

    def is_paused(self) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM autonomy_runtime_state WHERE key = 'paused'"
            ).fetchone()
        return bool(row and row["value"] == "1")

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO autonomy_runtime_state (key, value) VALUES ('paused', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("1" if paused else "0",),
            )
            self._conn.commit()

    def _finish_event(self, event_id, status, summary, notification, error) -> None:
        now = self._iso(datetime.now(timezone.utc))
        with self._lock:
            self._conn.execute(
                """
                UPDATE integration_events SET status = ?, outcome_summary = ?,
                    notification_json = ?, error = ?, completed_at = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (
                    status, summary,
                    self._json(notification) if notification is not None else None,
                    error, now, now, event_id,
                ),
            )
            self._conn.commit()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS integration_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    session_id TEXT,
                    payload_json TEXT NOT NULL,
                    attachments_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    correlation_id TEXT,
                    causation_id TEXT,
                    root_event_id TEXT NOT NULL,
                    deduplication_key TEXT,
                    replay_policy TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    processing_started_at TEXT,
                    completed_at TEXT,
                    outcome_summary TEXT,
                    notification_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_integration_events_pending
                    ON integration_events(status, priority, occurred_at);
                CREATE INDEX IF NOT EXISTS idx_integration_events_session
                    ON integration_events(session_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_integration_events_root
                    ON integration_events(root_event_id);

                CREATE TABLE IF NOT EXISTS integration_operations (
                    invocation_id TEXT PRIMARY KEY,
                    capability TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    event_id TEXT,
                    root_event_id TEXT,
                    causation_id TEXT,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS autonomy_runtime_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def _event_record(self, row: sqlite3.Row) -> EventRecord:
        attachments_payload = json.loads(row["attachments_json"] or "[]")
        from app.integrations import EventAttachmentRef

        event = IntegrationEvent(
            event=EventId.parse(row["event_type"]),
            payload=json.loads(row["payload_json"]),
            session_id=row["session_id"],
            event_id=row["event_id"],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            correlation_id=row["correlation_id"],
            causation_id=row["causation_id"],
            root_event_id=row["root_event_id"],
            deduplication_key=row["deduplication_key"],
            attachments=tuple(EventAttachmentRef(**item) for item in attachments_payload),
        )
        notification = json.loads(row["notification_json"]) if row["notification_json"] else None
        return EventRecord(
            event=event, status=row["status"], priority=int(row["priority"]),
            attempts=int(row["attempts"]), outcome_summary=row["outcome_summary"],
            notification=notification, error=row["error"],
        )

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    @staticmethod
    def _iso(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
