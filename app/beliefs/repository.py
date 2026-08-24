from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from app.beliefs.models import (
    BeliefMutation,
    BeliefRecord,
    CandidateOperation,
    EpistemicStatus,
    SubjectKind,
    VisibilityPolicy,
)
from app.storage.database import initialize_belief_schema


class StaleBeliefObservation(ValueError):
    pass


class BeliefRepository:
    """SQLite belief projection using an independent connection per operation."""

    def __init__(
        self,
        db,
        *,
        legacy_local_human_id: str = "local-human",
        legacy_local_human_name: str = "You",
    ):
        configured_path = str(db.path)
        self._uri = configured_path == ":memory:"
        self._path = (
            f"file:belief-{uuid.uuid4()}?mode=memory&cache=shared"
            if self._uri
            else configured_path
        )
        self._anchor = self._open_connection() if self._uri else None
        with self._connection() as conn:
            initialize_belief_schema(
                conn,
                legacy_local_human_id=legacy_local_human_id,
                legacy_local_human_name=legacy_local_human_name,
            )

    def close(self) -> None:
        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None

    def has_application(
        self,
        owner_agent_id: str,
        source_message_id: int,
        extractor_version: str,
    ) -> bool:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM belief_applications
                WHERE owner_agent_id = ? AND source_message_id = ? AND extractor_version = ?
                """,
                (owner_agent_id, source_message_id, extractor_version),
            ).fetchone()
        return row is not None

    def get_visible(
        self,
        owner_agent_id: str,
        session_id: str,
        *,
        now: datetime | None = None,
    ) -> list[BeliefRecord]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM beliefs
                WHERE owner_agent_id = ? AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > ?)
                  AND (
                    visibility = 'AGENT_CURRENT'
                    OR (visibility = 'SESSION_CURRENT' AND scope_session_id = ?)
                  )
                ORDER BY subject_id ASC, predicate ASC, epistemic_status ASC,
                         source_sender_id ASC, visibility ASC, belief_id ASC
                """,
                (owner_agent_id, self._iso(now), session_id),
            ).fetchall()
        return [self._record(row) for row in rows]

    # Backward-compatible repository spelling; limiting belongs in the snapshot layer.
    def get_active(
        self,
        owner_agent_id: str,
        session_id: str,
        *,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[BeliefRecord]:
        records = self.get_visible(owner_agent_id, session_id, now=now)
        return records if limit is None else records[:max(0, int(limit))]

    def get_by_id(self, belief_id: str) -> BeliefRecord | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM beliefs WHERE belief_id = ?",
                (belief_id,),
            ).fetchone()
        return self._record(row) if row else None

    def apply_mutations(
        self,
        *,
        owner_agent_id: str,
        source_message_id: int,
        extractor_version: str,
        mutations: list[BeliefMutation],
        now: datetime,
    ) -> bool:
        observed_at_iso = self._iso(now)
        with self._connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    """
                    SELECT 1 FROM belief_applications
                    WHERE owner_agent_id = ? AND source_message_id = ? AND extractor_version = ?
                    """,
                    (owner_agent_id, source_message_id, extractor_version),
                ).fetchone()
                if existing is not None:
                    conn.rollback()
                    return False

                for mutation in mutations:
                    if mutation.operation == CandidateOperation.INVALIDATE:
                        cursor = conn.execute(
                            """
                            UPDATE beliefs
                            SET status = 'invalidated', source_session_id = ?,
                                source_message_id = ?, source_observed_at = ?,
                                source_sender_id = ?, source_sender_display_name = ?,
                                source_sender_type = ?, source_input_source = ?,
                                evidence_excerpt = ?, revision = revision + 1, updated_at = ?
                            WHERE belief_id = ? AND owner_agent_id = ? AND status = 'active'
                              AND source_sender_id = ? AND epistemic_status = ?
                              AND (
                                source_observed_at < ?
                                OR (source_observed_at = ? AND source_message_id < ?)
                              )
                            """,
                            (
                                mutation.source_session_id,
                                source_message_id,
                                observed_at_iso,
                                mutation.source_sender_id,
                                mutation.source_sender_display_name,
                                mutation.source_sender_type,
                                mutation.source_input_source,
                                mutation.evidence_excerpt,
                                observed_at_iso,
                                mutation.belief_id,
                                owner_agent_id,
                                mutation.source_sender_id,
                                mutation.epistemic_status.value,
                                observed_at_iso,
                                observed_at_iso,
                                source_message_id,
                            ),
                        )
                        if cursor.rowcount != 1:
                            self._raise_stale_or_inactive(
                                conn, mutation.belief_id, owner_agent_id, observed_at_iso
                            )
                        continue

                    scope_session_id = (
                        mutation.source_session_id
                        if mutation.visibility == VisibilityPolicy.SESSION_CURRENT
                        else ""
                    )
                    value_json = json.dumps(
                        mutation.value,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    expires_at = self._iso(mutation.expires_at) if mutation.expires_at else None

                    if mutation.operation == CandidateOperation.UPDATE:
                        cursor = conn.execute(
                            """
                            UPDATE beliefs
                            SET source_session_id = ?, value_json = ?, status = 'active',
                                expires_at = ?, source_message_id = ?, source_observed_at = ?,
                                source_sender_display_name = ?, source_sender_type = ?,
                                source_input_source = ?,
                                evidence_excerpt = ?, revision = revision + 1, updated_at = ?
                            WHERE belief_id = ? AND owner_agent_id = ? AND status = 'active'
                              AND visibility = ? AND scope_session_id = ?
                              AND source_sender_id = ? AND epistemic_status = ?
                              AND (
                                source_observed_at < ?
                                OR (source_observed_at = ? AND source_message_id < ?)
                              )
                            """,
                            (
                                mutation.source_session_id,
                                value_json,
                                expires_at,
                                source_message_id,
                                observed_at_iso,
                                mutation.source_sender_display_name,
                                mutation.source_sender_type,
                                mutation.source_input_source,
                                mutation.evidence_excerpt,
                                observed_at_iso,
                                mutation.belief_id,
                                owner_agent_id,
                                mutation.visibility.value,
                                scope_session_id,
                                mutation.source_sender_id,
                                mutation.epistemic_status.value,
                                observed_at_iso,
                                observed_at_iso,
                                source_message_id,
                            ),
                        )
                        if cursor.rowcount != 1:
                            self._raise_stale_or_inactive(
                                conn, mutation.belief_id, owner_agent_id, observed_at_iso
                            )
                        continue

                    if mutation.operation not in {
                        CandidateOperation.ASSERT,
                        CandidateOperation.CREATE,
                    }:
                        raise ValueError("Unsupported belief mutation operation")

                    cursor = conn.execute(
                        """
                        INSERT INTO beliefs (
                            belief_id, owner_agent_id, visibility, scope_session_id,
                            source_session_id, subject_id, subject_kind, subject_display_name,
                            predicate, value_json, confidence,
                            status, expires_at, source_message_id, source_observed_at,
                            source_sender_id, source_sender_display_name, source_sender_type,
                            source_input_source, epistemic_status,
                            evidence_excerpt, revision, created_at, updated_at
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 'active', ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, 1, ?, ?
                        )
                        ON CONFLICT (
                            owner_agent_id, visibility, scope_session_id, subject_id, predicate,
                            epistemic_status, source_sender_id
                        ) DO UPDATE SET
                            source_session_id = excluded.source_session_id,
                            subject_kind = excluded.subject_kind,
                            subject_display_name = excluded.subject_display_name,
                            value_json = excluded.value_json,
                            confidence = excluded.confidence,
                            status = 'active',
                            expires_at = excluded.expires_at,
                            source_message_id = excluded.source_message_id,
                            source_observed_at = excluded.source_observed_at,
                            source_sender_display_name = excluded.source_sender_display_name,
                            source_sender_type = excluded.source_sender_type,
                            source_input_source = excluded.source_input_source,
                            evidence_excerpt = excluded.evidence_excerpt,
                            revision = beliefs.revision + 1,
                            updated_at = excluded.updated_at
                        WHERE beliefs.source_observed_at < excluded.source_observed_at
                           OR (
                             beliefs.source_observed_at = excluded.source_observed_at
                             AND beliefs.source_message_id < excluded.source_message_id
                           )
                        """,
                        (
                            str(uuid.uuid4()),
                            owner_agent_id,
                            mutation.visibility.value,
                            scope_session_id,
                            mutation.source_session_id,
                            mutation.subject_id,
                            mutation.subject_kind.value,
                            mutation.subject_display_name,
                            mutation.predicate,
                            value_json,
                            expires_at,
                            source_message_id,
                            observed_at_iso,
                            mutation.source_sender_id,
                            mutation.source_sender_display_name,
                            mutation.source_sender_type,
                            mutation.source_input_source,
                            mutation.epistemic_status.value,
                            mutation.evidence_excerpt,
                            observed_at_iso,
                            observed_at_iso,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise StaleBeliefObservation(
                            "Belief mutation was rejected because a newer observation is stored"
                        )

                conn.execute(
                    """
                    INSERT INTO belief_applications (
                        owner_agent_id, source_message_id, extractor_version, applied_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (owner_agent_id, source_message_id, extractor_version, observed_at_iso),
                )
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise

    def delete_session(self, owner_agent_id: str, session_id: str) -> int:
        """Delete only beliefs whose visibility is scoped to the deleted session."""
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                DELETE FROM beliefs
                WHERE owner_agent_id = ?
                  AND visibility = 'SESSION_CURRENT'
                  AND scope_session_id = ?
                """,
                (owner_agent_id, session_id),
            )
            conn.commit()
        return cursor.rowcount

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._path,
            uri=self._uri,
            check_same_thread=False,
            timeout=5.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _connection(self):
        conn = self._open_connection()
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _raise_stale_or_inactive(conn, belief_id, owner_agent_id, observed_at_iso):
        row = conn.execute(
            """
            SELECT status, source_observed_at FROM beliefs
            WHERE belief_id = ? AND owner_agent_id = ?
            """,
            (belief_id, owner_agent_id),
        ).fetchone()
        if row is not None and row["source_observed_at"] >= observed_at_iso:
            raise StaleBeliefObservation(
                "Belief mutation was rejected because a newer observation is stored"
            )
        raise ValueError("Belief target is no longer active or its scope changed")

    @staticmethod
    def _iso(value: datetime) -> str:
        if value.tzinfo is None:
            raise ValueError("Belief timestamps must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _record(row) -> BeliefRecord:
        return BeliefRecord(
            belief_id=row["belief_id"],
            owner_agent_id=row["owner_agent_id"],
            visibility=VisibilityPolicy(row["visibility"]),
            source_session_id=row["source_session_id"],
            subject_id=row["subject_id"],
            subject_kind=SubjectKind(row["subject_kind"]),
            subject_display_name=row["subject_display_name"],
            predicate=row["predicate"],
            value=json.loads(row["value_json"]),
            confidence=float(row["confidence"]),
            status=row["status"],
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            source_message_id=int(row["source_message_id"]),
            source_observed_at=datetime.fromisoformat(row["source_observed_at"]),
            source_sender_id=row["source_sender_id"],
            source_sender_display_name=row["source_sender_display_name"],
            source_sender_type=row["source_sender_type"],
            source_input_source=row["source_input_source"],
            epistemic_status=EpistemicStatus(row["epistemic_status"]),
            evidence_excerpt=row["evidence_excerpt"],
            revision=int(row["revision"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
