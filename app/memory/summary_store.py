from app.storage.database import Database


class SummaryStore:
    def __init__(self, db: Database):
        self.db = db

    def get(self, session_id: str) -> str | None:
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT summary
            FROM conversation_summary
            WHERE session_id = ?
            """,
            (session_id,)
        )
        row = cursor.fetchone()
        return row["summary"] if row else None
    
    def set(self, session_id: str, summary: str) -> None:
        self._upsert(session_id, summary)

    def _upsert(self, session_id: str, summary: str) -> None:
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversation_summary (session_id, summary)
            VALUES (?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET
                summary = excluded.summary,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, summary)
        )
        self.db.conn.commit()
