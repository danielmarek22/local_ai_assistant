import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path: str = "data/assistant.db"):
        Path("data").mkdir(exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            name TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER,
            summary_text TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES chat_history(id) ON DELETE CASCADE
        )
        """)
        attachment_columns = {
            row["name"]
            for row in cursor.execute("PRAGMA table_info(chat_attachments)")
        }
        if "summary_text" not in attachment_columns:
            cursor.execute("ALTER TABLE chat_attachments ADD COLUMN summary_text TEXT")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_attachments_message_id ON chat_attachments(message_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_attachments_session_id ON chat_attachments(session_id)"
        )

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            content TEXT NOT NULL,
            importance INTEGER DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summary (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        self.conn.commit()
