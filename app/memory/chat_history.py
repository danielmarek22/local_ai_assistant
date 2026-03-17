import uuid
import time
from app.storage.database import Database
from app.storage.vector_store import VectorStore

class ChatHistoryStore:
    def __init__(self, db: Database, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store
        self.collection = self.vector_store.episodic_collection

    def add(self, session_id: str, role: str, content: str):
        msg_id = str(uuid.uuid4())
        current_time = time.time()
        
        # 1. Save to SQLite (Maintains the exact UI sequence and sliding window)
        cursor = self.db.conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        self.db.conn.commit()

        # 2. Save to Vector Store (Allows the AI to "feel" past contexts)
        # We include the role in the document so the AI knows who said what
        vector_doc = f"{role.upper()}: {content}"
        
        self.collection.add(
            ids=[msg_id],
            documents=[vector_doc],
            metadatas=[{
                "session_id": session_id,
                "role": role,
                "timestamp": current_time
            }]
        )

    def search_past_conversations(self, query: str, current_session: str, limit: int = 4) -> list[str]:
        """
        Retrieves relevant past messages, explicitly filtering OUT the current 
        active session (since the sliding window already handles the current session).
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where={"session_id": {"$ne": current_session}} # Filter out current chat
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []
            
        return results["documents"][0]

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
