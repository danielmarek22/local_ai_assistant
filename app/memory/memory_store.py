import uuid
import logging

from app.logging import trace_event
from app.storage.database import Database
from app.storage.vector_store import VectorStore

logger = logging.getLogger("memory_store")


class MemoryStore:
    def __init__(self, db: Database, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store
        self.collection = self.vector_store.semantic_collection

    def add(self, content: str, category: str = "general", importance: int = 1) -> None:
        mem_id = str(uuid.uuid4())
        trace_event(
            "memory_store",
            "memory_saved",
            payload={
                "id": mem_id,
                "content": content,
                "category": category,
                "importance": importance,
            },
        )
        
        # 1. Save to SQLite with shared UUID
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory (id, category, content, importance) 
            VALUES (?, ?, ?, ?)
            """,
            (mem_id, category, content, importance),
        )
        self.db.conn.commit()

        # 2. Save to Vector Store using the SAME UUID
        self.collection.add(
            ids=[mem_id],
            documents=[content],
            metadatas=[{"category": category, "importance": importance}]
        )

    def get_all(self, limit: int = 20) -> list[dict]:
        """
        Retrieves recent memories for the UI/frontend.
        Returns full dicts so the UI can display timestamps!
        """
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT id, content, importance, created_at, last_accessed_at
            FROM memory
            ORDER BY importance DESC, last_accessed_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_relevant(self, query: str, limit: int = 3) -> list[str]:
        """
        True semantic search using CPU embeddings for the Orchestrator.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        if not results["documents"] or not results["documents"][0]:
            logger.debug("Semantic memory search returned no results")
            trace_event(
                "memory_store",
                "semantic_query_result",
                payload={"query": query, "limit": limit, "documents": []},
            )
            return []

        retrieved_ids = results["ids"][0]
        documents = results["documents"][0]

        # Bump the access timestamp in SQLite for the memories we just pulled
        if retrieved_ids:
            self._touch_memories(retrieved_ids)

        trace_event(
            "memory_store",
            "semantic_query_result",
            payload={"query": query, "limit": limit, "documents": documents},
        )
        return documents

    def _touch_memories(self, memory_ids: list[str]):
        """Helper to update the last_accessed_at timestamp."""
        if not memory_ids:
            return
            
        cursor = self.db.conn.cursor()
        # Create a parameter placeholder list like (?, ?, ?)
        placeholders = ",".join(["?"] * len(memory_ids))
        
        cursor.execute(
            f"""
            UPDATE memory 
            SET last_accessed_at = CURRENT_TIMESTAMP 
            WHERE id IN ({placeholders})
            """,
            memory_ids
        )
        self.db.conn.commit()
        logger.debug("Updated last_accessed_at for %d memories", len(memory_ids))