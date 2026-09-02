import sqlite3
from pathlib import Path


def _create_beliefs_table(conn: sqlite3.Connection, table_name: str = "beliefs") -> None:
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        belief_id TEXT PRIMARY KEY,
        owner_agent_id TEXT NOT NULL,
        visibility TEXT NOT NULL CHECK (
            visibility IN ('AGENT_CURRENT', 'SESSION_CURRENT')
        ),
        scope_session_id TEXT NOT NULL DEFAULT '',
        source_session_id TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        subject_kind TEXT NOT NULL CHECK (
            subject_kind IN ('PERSON', 'AGENT', 'WORLD', 'ENVIRONMENT', 'PROJECT', 'OTHER')
        ),
        subject_display_name TEXT NOT NULL,
        predicate TEXT NOT NULL,
        value_json TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        status TEXT NOT NULL CHECK (status IN ('active', 'invalidated')),
        expires_at TEXT,
        source_message_id INTEGER NOT NULL,
        source_observed_at TEXT NOT NULL,
        source_sender_id TEXT NOT NULL,
        source_sender_display_name TEXT NOT NULL,
        source_sender_type TEXT NOT NULL,
        source_input_source TEXT NOT NULL,
        epistemic_status TEXT NOT NULL CHECK (
            epistemic_status IN ('SELF_REPORT', 'ATTRIBUTED_CLAIM')
        ),
        evidence_excerpt TEXT,
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (
            (visibility = 'AGENT_CURRENT' AND scope_session_id = '')
            OR (visibility = 'SESSION_CURRENT' AND length(scope_session_id) > 0)
        ),
        UNIQUE (
            owner_agent_id, visibility, scope_session_id, subject_id, predicate,
            epistemic_status, source_sender_id
        )
    )
    """)


def initialize_belief_schema(
    conn: sqlite3.Connection,
    *,
    legacy_local_human_id: str = "local-human",
    legacy_local_human_name: str = "You",
) -> None:
    existing = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'beliefs'"
    ).fetchone()
    if existing is None:
        _create_beliefs_table(conn)
    else:
        belief_columns = {
            row["name"] if isinstance(row, sqlite3.Row) else row[1]
            for row in conn.execute("PRAGMA table_info(beliefs)")
        }
        if "subject_id" not in belief_columns:
            _migrate_legacy_beliefs(
                conn,
                legacy_local_human_id=legacy_local_human_id,
                legacy_local_human_name=legacy_local_human_name,
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


def _migrate_legacy_beliefs(
    conn: sqlite3.Connection,
    *,
    legacy_local_human_id: str,
    legacy_local_human_name: str,
) -> None:
    """Atomically rebuild the legacy user/world/environment belief projection."""
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DROP TABLE IF EXISTS beliefs_v2")
        _create_beliefs_table(conn, "beliefs_v2")
        history_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chat_history'"
        ).fetchone() is not None
        rows = conn.execute("SELECT * FROM beliefs ORDER BY belief_id").fetchall()
        for row in rows:
            row = dict(row)
            sender = None
            if history_exists:
                sender = conn.execute(
                    """
                    SELECT sender_id, sender_display_name, sender_type, input_source
                    FROM chat_history WHERE id = ?
                    """,
                    (row["source_message_id"],),
                ).fetchone()
            sender = dict(sender) if sender is not None else {}
            source_sender_id = sender.get("sender_id") or legacy_local_human_id
            source_sender_name = sender.get("sender_display_name") or legacy_local_human_name
            source_sender_type = sender.get("sender_type") or "human"
            source_input_source = sender.get("input_source") or "local_text"
            legacy_subject = row["subject"]
            if legacy_subject == "user":
                subject_id = source_sender_id
                subject_kind = "AGENT" if source_sender_type == "external_agent" else "PERSON"
                subject_display_name = source_sender_name
                epistemic_status = "SELF_REPORT"
            elif legacy_subject == "world":
                subject_id, subject_kind, subject_display_name = (
                    "entity:world", "WORLD", "World"
                )
                epistemic_status = "ATTRIBUTED_CLAIM"
            else:
                subject_id, subject_kind, subject_display_name = (
                    "entity:environment:default", "ENVIRONMENT", "Environment"
                )
                epistemic_status = "ATTRIBUTED_CLAIM"
            conn.execute(
                """
                INSERT INTO beliefs_v2 (
                    belief_id, owner_agent_id, visibility, scope_session_id,
                    source_session_id, subject_id, subject_kind, subject_display_name,
                    predicate, value_json, confidence, status, expires_at,
                    source_message_id, source_observed_at, source_sender_id,
                    source_sender_display_name, source_sender_type, source_input_source,
                    epistemic_status, evidence_excerpt, revision, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    row["belief_id"], row["owner_agent_id"], row["visibility"],
                    row["scope_session_id"], row["origin_session_id"], subject_id,
                    subject_kind, subject_display_name, row["predicate"], row["value_json"],
                    row["confidence"], row["status"], row["expires_at"],
                    row["source_message_id"], row.get("source_observed_at") or row["updated_at"],
                    source_sender_id, source_sender_name, source_sender_type,
                    source_input_source, epistemic_status, row["evidence_excerpt"],
                    row["revision"], row["created_at"], row["updated_at"],
                ),
            )
        conn.execute("DROP TABLE beliefs")
        conn.execute("ALTER TABLE beliefs_v2 RENAME TO beliefs")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class Database:
    def __init__(
        self,
        path: str = "data/assistant.db",
        *,
        legacy_local_human_id: str = "local-human",
        legacy_local_human_name: str = "You",
    ):
        self.path = path
        self.legacy_local_human_id = legacy_local_human_id
        self.legacy_local_human_name = legacy_local_human_name
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
        history_columns = {
            row["name"]
            for row in cursor.execute("PRAGMA table_info(chat_history)")
        }
        for column_name in (
            "sender_id",
            "sender_display_name",
            "sender_type",
            "input_source",
        ):
            if column_name not in history_columns:
                cursor.execute(f"ALTER TABLE chat_history ADD COLUMN {column_name} TEXT")
        if "retry_error" not in history_columns:
            cursor.execute("ALTER TABLE chat_history ADD COLUMN retry_error TEXT")
        if "retry_attempts" not in history_columns:
            cursor.execute(
                "ALTER TABLE chat_history ADD COLUMN retry_attempts INTEGER NOT NULL DEFAULT 0"
            )
        if "excluded_from_context" not in history_columns:
            cursor.execute(
                "ALTER TABLE chat_history ADD COLUMN excluded_from_context INTEGER NOT NULL DEFAULT 0"
            )

        # Preserve old deterministic fallbacks as diagnostics, but attach their
        # failure state to the preceding user turn and keep the fallback prose
        # out of both the UI and future model context.
        legacy_fallback = "I'm sorry, I lost my train of thought. Could you repeat that?"
        legacy_rows = cursor.execute(
            """
            SELECT id, session_id FROM chat_history
            WHERE role = 'assistant' AND content = ? AND excluded_from_context = 0
            """,
            (legacy_fallback,),
        ).fetchall()
        for fallback_row in legacy_rows:
            user_row = cursor.execute(
                """
                SELECT id FROM chat_history
                WHERE session_id = ? AND role = 'user' AND id < ?
                ORDER BY id DESC LIMIT 1
                """,
                (fallback_row["session_id"], fallback_row["id"]),
            ).fetchone()
            if user_row is not None:
                cursor.execute(
                    """
                    UPDATE chat_history
                    SET retry_error = COALESCE(retry_error, ?),
                        retry_attempts = MAX(retry_attempts, 1)
                    WHERE id = ?
                    """,
                    ("Astra couldn't finish this response.", user_row["id"]),
                )
            cursor.execute(
                "UPDATE chat_history SET excluded_from_context = 1 WHERE id = ?",
                (fallback_row["id"],),
            )

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('direct', 'manual_group')),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
        initialize_belief_schema(
            self.conn,
            legacy_local_human_id=self.legacy_local_human_id,
            legacy_local_human_name=self.legacy_local_human_name,
        )
