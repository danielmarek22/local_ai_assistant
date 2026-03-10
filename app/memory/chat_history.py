from app.storage.database import Database


class ChatHistoryStore:
    def __init__(self, db: Database):
        self.db = db

    def add(self, session_id: str, role: str, content: str):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_history (session_id, role, content)
            VALUES (?, ?, ?)
            """,
            (session_id, role, content)
        )
        self.db.conn.commit()

    def get_recent(self, session_id: str, limit: int = 10):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT role, content
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit)
        )
        rows = cursor.fetchall()
        return list(reversed(rows))

    def get_all(self, session_id: str):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT role, content, timestamp
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )
        return cursor.fetchall()

    def list_sessions(self):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT
                ch.session_id,
                MIN(ch.timestamp) AS started_at,
                MAX(ch.timestamp) AS updated_at,
                COUNT(*) AS message_count,
                (
                    SELECT ch2.content
                    FROM chat_history ch2
                    WHERE ch2.session_id = ch.session_id
                    ORDER BY ch2.id ASC
                    LIMIT 1
                ) AS preview
            FROM chat_history ch
            GROUP BY ch.session_id
            ORDER BY updated_at DESC, ch.session_id DESC
            """
        )
        return cursor.fetchall()

    def delete_session(self, session_id: str) -> int:
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            DELETE FROM chat_history
            WHERE session_id = ?
            """,
            (session_id,)
        )
        self.db.conn.commit()
        return cursor.rowcount
