from app.logging import trace_event
from app.storage.database import Database
from app.storage.session_lifecycle import require_writable_session


class SummaryStore:
    def __init__(self, db: Database):
        self.db = db

    def get(self, session_id: str) -> tuple[str, int] | None:
        """Returns a tuple of (summary_text, last_turn_count) or None."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT summary, last_turn_count
                FROM conversation_summary
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        result = (row["summary"], row["last_turn_count"]) if row else None
        trace_event(
            "summary_store",
            "summary_get",
            session_id=session_id,
            payload={"result": result},
        )
        return result
    
    def set(self, session_id: str, summary: str, last_turn_count: int) -> None:
        self._upsert(session_id, summary, last_turn_count)

    def _upsert(self, session_id: str, summary: str, last_turn_count: int) -> None:
        trace_event(
            "summary_store",
            "summary_set",
            session_id=session_id,
            payload={"summary": summary, "last_turn_count": last_turn_count},
        )
        with self.db.transaction() as conn:
            require_writable_session(conn, session_id)
            conn.execute(
                """
                INSERT INTO conversation_summary (session_id, summary, last_turn_count)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id)
                DO UPDATE SET
                    summary = excluded.summary,
                    last_turn_count = excluded.last_turn_count,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, summary, last_turn_count),
            )

    def delete(self, session_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                DELETE FROM conversation_summary
                WHERE session_id = ?
                """,
                (session_id,),
            )
