"""Durable session deletion fences shared by canonical stores and the event journal."""

from sqlite3 import Connection

from app.core.session_ids import SessionDeletedError


def initialize_session_lifecycle(conn: Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS deleted_sessions (
            session_id TEXT PRIMARY KEY,
            deleted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)


def is_session_deleted(conn: Connection, session_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM deleted_sessions WHERE session_id = ?", (session_id,),
    ).fetchone() is not None


def require_writable_session(conn: Connection, session_id: str) -> None:
    if is_session_deleted(conn, session_id):
        raise SessionDeletedError()
