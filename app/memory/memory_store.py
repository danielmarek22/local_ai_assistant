from app.storage.database import Database


class MemoryStore:
    def __init__(self, db: Database):
        self.db = db

    def add(self, content: str, key: str | None = None, importance: int = 1):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory (key, content, importance)
            VALUES (?, ?, ?)
            """,
            (key, content, importance)
        )
        self.db.conn.commit()

    def get_all(self, limit: int = 20):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT content
            FROM memory
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        return [row["content"] for row in cursor.fetchall()]
