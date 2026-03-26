from app.storage.database import Database

class SummaryStore:
    def __init__(self, db: Database):
        self.db = db

    def get(self, session_id: str) -> tuple[str, int] | None:
        """Returns a tuple of (summary_text, last_turn_count) or None."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT summary, last_turn_count
            FROM conversation_summary
            WHERE session_id = ?
            """,
            (session_id,)
        )
        row = cursor.fetchone()
        return (row["summary"], row["last_turn_count"]) if row else None
    
    def set(self, session_id: str, summary: str, last_turn_count: int) -> None:
        self._upsert(session_id, summary, last_turn_count)

    def _upsert(self, session_id: str, summary: str, last_turn_count: int) -> None:
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversation_summary (session_id, summary, last_turn_count)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET
                summary = excluded.summary,
                last_turn_count = excluded.last_turn_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, summary, last_turn_count)
        )
        self.db.conn.commit()

    def delete(self, session_id: str) -> None:
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            DELETE FROM conversation_summary
            WHERE session_id = ?
            """,
            (session_id,)
        )
        self.db.conn.commit()