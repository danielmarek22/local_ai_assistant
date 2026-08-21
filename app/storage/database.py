import sqlite3
from pathlib import Path


def initialize_belief_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS beliefs (
        belief_id TEXT PRIMARY KEY,
        owner_agent_id TEXT NOT NULL,
        visibility TEXT NOT NULL CHECK (
            visibility IN ('AGENT_CURRENT', 'SESSION_CURRENT')
        ),
        scope_session_id TEXT NOT NULL DEFAULT '',
        origin_session_id TEXT NOT NULL,
        subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        value_json TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        status TEXT NOT NULL CHECK (status IN ('active', 'invalidated')),
        expires_at TEXT,
        source_message_id INTEGER NOT NULL,
        source_observed_at TEXT NOT NULL,
        evidence_excerpt TEXT,
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (
            (visibility = 'AGENT_CURRENT' AND scope_session_id = '')
            OR (visibility = 'SESSION_CURRENT' AND length(scope_session_id) > 0)
        ),
        UNIQUE (
            owner_agent_id, visibility, scope_session_id, subject, predicate
        )
    )
    """)
    belief_columns = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in conn.execute("PRAGMA table_info(beliefs)")
    }
    if "source_observed_at" not in belief_columns:
        conn.execute("ALTER TABLE beliefs ADD COLUMN source_observed_at TEXT")
        conn.execute(
            "UPDATE beliefs SET source_observed_at = updated_at WHERE source_observed_at IS NULL"
        )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_beliefs_active_scope
        ON beliefs(owner_agent_id, status, visibility, scope_session_id, expires_at)
        """
    )
    conn.execute("""
    CREATE TABLE IF NOT EXISTS belief_applications (
        owner_agent_id TEXT NOT NULL,
        source_message_id INTEGER NOT NULL,
        extractor_version TEXT NOT NULL,
        applied_at TEXT NOT NULL,
        PRIMARY KEY (owner_agent_id, source_message_id, extractor_version)
    )
    """)
    conn.commit()


class Database:
    def __init__(self, path: str = "data/assistant.db"):
        self.path = path
        Path("data").mkdir(exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
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

        # UPDATED: id is TEXT, added last_accessed_at
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            category TEXT,
            content TEXT NOT NULL,
            importance INTEGER DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summary (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            last_turn_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        self.conn.commit()
        initialize_belief_schema(self.conn)
