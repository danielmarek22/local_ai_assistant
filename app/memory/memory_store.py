import uuid
import logging

from app.logging import trace_event
from app.storage.database import Database
from app.storage.vector_store import VectorStore

logger = logging.getLogger("memory_store")


class MemoryIndexSyncError(RuntimeError):
    """A canonical SQLite mutation succeeded but its vector update failed."""

    def __init__(self, operation: str, memory_ids: list[str], canonical_changes: int):
        self.operation = operation
        self.memory_ids = memory_ids
        self.canonical_changes = canonical_changes
        super().__init__(
            f"Memory {operation} indexing failed after {canonical_changes} canonical change(s)"
        )


class MemoryStore:
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        *,
        max_distance: float = 0.70,
        fallback_max_distance: float = 0.85,
        fallback_limit: int = 2,
    ):
        self.db = db
        self.vector_store = vector_store
        self.collection = self.vector_store.semantic_collection
        self.max_distance = float(max_distance)
        self.fallback_max_distance = float(fallback_max_distance)
        self.fallback_limit = max(1, int(fallback_limit))

    def add(self, content: str, category: str = "general", importance: int = 1) -> str:
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
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO memory (id, category, content, importance)
                VALUES (?, ?, ?, ?)
                """,
                (mem_id, category, content, importance),
            )

        # 2. Idempotently index the canonical row using the SAME UUID.
        try:
            self.collection.upsert(
                ids=[mem_id],
                documents=[content],
                metadatas=[{"category": category, "importance": importance}],
            )
        except Exception as exc:
            raise MemoryIndexSyncError("upsert", [mem_id], 1) from exc
        return mem_id

    def apply_consolidation(self, delete_ids: list[str], new_memories: list[dict]) -> dict:
        """Commit the complete reflection in SQLite before syncing derived vectors."""
        delete_ids = list(dict.fromkeys(delete_ids))
        additions = [{**memory, "id": str(uuid.uuid4())} for memory in new_memories]
        with self.db.transaction() as conn:
            for memory_id in delete_ids:
                cursor = conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
                if cursor.rowcount != 1:
                    raise ValueError("Reflection source memory changed; run reflection again")
            for memory in additions:
                conn.execute(
                    "INSERT INTO memory (id, category, content, importance) VALUES (?, ?, ?, ?)",
                    (memory["id"], memory["category"], memory["content"], memory["importance"]),
                )

        errors = []
        # Try both operations even if one fails. Canonical replacements already
        # exist and reconcile_index can repair either kind of divergence later.
        if additions:
            try:
                self.collection.upsert(
                    ids=[memory["id"] for memory in additions],
                    documents=[memory["content"] for memory in additions],
                    metadatas=[{"category": memory["category"], "importance": memory["importance"]}
                               for memory in additions],
                )
            except Exception:
                logger.exception("Reflection committed, but replacement indexing failed")
                errors.append("Replacement memories could not be indexed")
        if delete_ids:
            try:
                self.collection.delete(ids=delete_ids)
            except Exception:
                logger.exception("Reflection committed, but obsolete vector cleanup failed")
                errors.append("Obsolete memory vectors could not be removed")
        return {
            "deleted_count": len(delete_ids),
            "created_ids": [memory["id"] for memory in additions],
            "index_sync_complete": not errors,
            "index_sync_errors": errors,
        }

    def reconcile_index(self, batch_size: int = 100) -> dict[str, int]:
        """Make the semantic index match canonical SQLite memory rows."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        rows = self.list_for_inspection()
        canonical_ids = {row["id"] for row in rows}
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            self.collection.upsert(
                ids=[row["id"] for row in batch],
                documents=[row["content"] for row in batch],
                metadatas=[
                    {
                        "category": row["category"] or "general",
                        "importance": row["importance"],
                    }
                    for row in batch
                ],
            )

        indexed_ids = set(self.collection.get(include=[])["ids"])
        orphaned_ids = sorted(indexed_ids - canonical_ids)
        if orphaned_ids:
            self.collection.delete(ids=orphaned_ids)

        report = {
            "canonical_count": len(rows),
            "upserted_count": len(rows),
            "removed_count": len(orphaned_ids),
        }
        logger.info("Reconciled semantic memory index: %s", report)
        return report

    def get_all(self, limit: int = 20) -> list[dict]:
        """
        Retrieves recent memories for the UI/frontend.
        Returns full dicts so the UI can display timestamps!
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, content, importance, created_at, last_accessed_at
                FROM memory
                ORDER BY importance DESC, last_accessed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_for_inspection(self) -> list[dict]:
        """Return every canonical SQLite memory row without retrieval side effects."""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, category, content, importance, created_at, last_accessed_at
                FROM memory
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_relevant(
        self,
        query: str,
        limit: int = 3,
        max_distance: float | None = None,
        fallback_max_distance: float | None = None,
        fallback_limit: int | None = None,
    ) -> list[str]:
        """
        True semantic search using CPU embeddings for the Orchestrator, 
        now with a strict similarity threshold.
        """
        strict_distance = self.max_distance if max_distance is None else float(max_distance)
        fallback_distance = (
            self.fallback_max_distance
            if fallback_max_distance is None
            else float(fallback_max_distance)
        )
        bounded_fallback_limit = min(
            limit,
            self.fallback_limit if fallback_limit is None else max(1, int(fallback_limit)),
        )
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

        candidates = list(zip(retrieved_ids, documents, distances))
        selected = [
            candidate for candidate in candidates if candidate[2] <= strict_distance
        ]
        selection_mode = "strict"
        if not selected:
            selected = [
                candidate for candidate in candidates if candidate[2] <= fallback_distance
            ][:bounded_fallback_limit]
            selection_mode = "fallback" if selected else "none"

        filtered_ids = [candidate[0] for candidate in selected]
        filtered_docs = [candidate[1] for candidate in selected]
        selected_ids = set(filtered_ids)
        for doc_id, doc, distance in candidates:
            if doc_id not in selected_ids:
                logger.debug(
                    "Discarded memory '%s...' (distance=%.3f, strict=%.3f, fallback=%.3f)",
                    doc[:30],
                    distance,
                    strict_distance,
                    fallback_distance,
                )

        if filtered_ids:
            self._touch_memories(filtered_ids)

        trace_event(
            "memory_store",
            "semantic_query_result",
            payload={
                "query": query,
                "limit": limit,
                "selection_mode": selection_mode,
                "strict_max_distance": strict_distance,
                "fallback_max_distance": fallback_distance,
                "fallback_limit": bounded_fallback_limit,
                "candidates": [
                    {
                        "id": doc_id,
                        "distance": distance,
                        "selected": doc_id in selected_ids,
                    }
                    for doc_id, _doc, distance in candidates
                ],
                "documents": filtered_docs,
            },
        )
        return filtered_docs

    def _touch_memories(self, memory_ids: list[str]):
        """Helper to update the last_accessed_at timestamp."""
        if not memory_ids:
            return
            
        # Create a parameter placeholder list like (?, ?, ?)
        placeholders = ",".join(["?"] * len(memory_ids))

        with self.db.transaction() as conn:
            conn.execute(
                f"""
                UPDATE memory
                SET last_accessed_at = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                """,
                memory_ids,
            )
        logger.debug("Updated last_accessed_at for %d memories", len(memory_ids))

    def get_stale(self, days_old: int = 14) -> list[dict]:
        """Fetches memories that haven't been accessed in X days."""
        # Using SQLite's built-in date math
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, category, content, importance, created_at, last_accessed_at
                FROM memory
                WHERE last_accessed_at <= datetime('now', '-{days_old} days')
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_memories(self, memory_ids: list[str]) -> int:
        """Deletes memories from both SQLite and the Vector DB."""
        if not memory_ids:
            return 0

        # 1. Delete from SQLite
        placeholders = ",".join(["?"] * len(memory_ids))
        with self.db.transaction() as conn:
            cursor = conn.execute(
                f"DELETE FROM memory WHERE id IN ({placeholders})",
                memory_ids,
            )
            deleted_count = cursor.rowcount

        # 2. Delete from ChromaDB. A reconciliation can complete this cleanup.
        try:
            self.collection.delete(ids=memory_ids)
        except Exception as exc:
            raise MemoryIndexSyncError("delete", memory_ids, deleted_count) from exc
        logger.info("Deleted %d stale memories", deleted_count)
        return deleted_count
