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

    def get_relevant(self, query: str, limit: int = 3, max_distance: float = 0.65) -> list[str]:
        """
        True semantic search using CPU embeddings for the Orchestrator, 
        now with a strict similarity threshold.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        if not results["documents"] or not results["documents"][0]:
            logger.debug("Semantic memory search returned no results")
            return []

        retrieved_ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else []

        filtered_docs = []
        filtered_ids = []

        # STRICT FILTERING: Drop anything with a distance > 0.55
        for doc_id, doc, distance in zip(retrieved_ids, documents, distances):
            if distance <= max_distance:
                filtered_docs.append(doc)
                filtered_ids.append(doc_id)
            else:
                logger.debug(f"Discarded memory '{doc[:30]}...' (Distance: {distance:.3f} > {max_distance})")

        if filtered_ids:
            self._touch_memories(filtered_ids)

        trace_event(
            "memory_store",
            "semantic_query_result",
            payload={"query": query, "limit": limit, "documents": filtered_docs},
        )
        return filtered_docs

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

    def get_stale(self, days_old: int = 14) -> list[dict]:
        """Fetches memories that haven't been accessed in X days."""
        cursor = self.db.conn.cursor()
        # Using SQLite's built-in date math
        cursor.execute(
            f"""
            SELECT id, category, content, importance, created_at, last_accessed_at
            FROM memory
            WHERE last_accessed_at <= datetime('now', '-{days_old} days')
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_memories(self, memory_ids: list[str]) -> int:
        """Deletes memories from both SQLite and the Vector DB."""
        if not memory_ids:
            return 0

        # 1. Delete from SQLite
        cursor = self.db.conn.cursor()
        placeholders = ",".join(["?"] * len(memory_ids))
        cursor.execute(
            f"DELETE FROM memory WHERE id IN ({placeholders})",
            memory_ids
        )
        deleted_count = cursor.rowcount
        self.db.conn.commit()

        # 2. Delete from ChromaDB
        self.collection.delete(ids=memory_ids)
        logger.info("Deleted %d stale memories", deleted_count)
        return deleted_count
