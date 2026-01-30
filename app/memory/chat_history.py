from storage.database import Database


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
