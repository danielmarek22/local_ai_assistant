import unittest
from unittest.mock import patch
from app.storage.database import Database
from app.memory.memory_store import MemoryStore


class FakeCollection:
    def __init__(self):
        self.ids = []
        self.docs = []
        self.distances = []
        self.deleted_ids = []

    def add(self, ids, documents, metadatas):
        self.ids.extend(ids)
        self.docs.extend(documents)
        self.distances.extend(metadata.get("distance", 0.2) for metadata in metadatas)

    def query(self, query_texts, n_results, where=None):
        # Fake search behavior: return everything we have up to n_results
        return {
            "ids": [self.ids[:n_results]] if self.ids else [[]],
            "documents": [self.docs[:n_results]] if self.docs else [[]],
            "distances": [self.distances[:n_results]] if self.distances else [[]],
        }

    def delete(self, ids=None, where=None):
        if ids:
            self.deleted_ids.extend(ids)
            filtered_records = [
                (mem_id, doc, distance)
                for mem_id, doc, distance in zip(self.ids, self.docs, self.distances)
                if mem_id not in ids
            ]
            self.ids = [record[0] for record in filtered_records]
            self.docs = [record[1] for record in filtered_records]
            self.distances = [record[2] for record in filtered_records]


class FakeVectorStore:
    def __init__(self):
        self.semantic_collection = FakeCollection()
        self.episodic_collection = FakeCollection()


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(path=":memory:")
        self.vector_store = FakeVectorStore()
        self.store = MemoryStore(self.db, self.vector_store)

    def test_add_saves_to_both_sqlite_and_vectordb(self):
        self.store.add("I like Python", category="tech", importance=3)

        # 1. Check SQLite
        sqlite_results = self.store.get_all(limit=10)
        self.assertEqual(len(sqlite_results), 1)
        self.assertEqual(sqlite_results[0]["content"], "I like Python")
        self.assertIsNotNone(sqlite_results[0]["created_at"])
        self.assertIsNotNone(sqlite_results[0]["last_accessed_at"])

        # 2. Check VectorDB mock collection
        self.assertEqual(len(self.vector_store.semantic_collection.docs), 1)
        self.assertEqual(self.vector_store.semantic_collection.docs[0], "I like Python")

    def test_get_relevant_queries_vectordb(self):
        self.store.add("Memory A")
        self.store.add("Memory B")

        # Our fake collection returns hits with distances under the threshold.
        results = self.store.get_relevant("find A", limit=2)
        
        self.assertEqual(len(results), 2)
        self.assertIn("Memory A", results)

    def test_get_relevant_filters_vectordb_results_by_distance(self):
        self.store.add("Relevant memory")
        self.store.add("Distant memory")
        self.vector_store.semantic_collection.distances = [0.2, 0.9]

        results = self.store.get_relevant("find relevant", limit=2, max_distance=0.65)

        self.assertEqual(results, ["Relevant memory"])

    def test_get_stale_with_zero_days_returns_all_memories(self):
        self.store.add("Memory A", category="general", importance=1)
        self.store.add("Memory B", category="general", importance=2)

        stale = self.store.get_stale(days_old=0)

        self.assertEqual(len(stale), 2)
        self.assertEqual({memory["content"] for memory in stale}, {"Memory A", "Memory B"})

    def test_delete_memories_removes_sqlite_and_vectordb_rows(self):
        self.store.add("Delete me", category="general", importance=1)
        memories = self.store.get_all(limit=10)
        self.assertEqual(len(memories), 1)
        memory_id = memories[0]["id"]

        deleted_count = self.store.delete_memories([memory_id])

        self.assertEqual(deleted_count, 1)
        self.assertEqual(self.store.get_all(limit=10), [])
        self.assertEqual(self.vector_store.semantic_collection.deleted_ids, [memory_id])

    def test_inspection_returns_actual_schema_in_stable_order_without_mutation(self):
        rows = [
            ("memory-a", "general", "Older", 1, "2026-08-25 09:00:00", "2026-08-25 10:00:00"),
            ("memory-b", "preference", "Same timestamp B", 3, "2026-08-26 09:00:00", "2026-08-26 10:00:00"),
            ("memory-c", None, "Same timestamp C", 2, "2026-08-26 09:00:00", "2026-08-26 11:00:00"),
        ]
        self.db.conn.executemany(
            """
            INSERT INTO memory (id, category, content, importance, created_at, last_accessed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.db.conn.commit()
        before = [tuple(row) for row in self.db.conn.execute(
            "SELECT id, category, content, importance, created_at, last_accessed_at FROM memory ORDER BY id"
        ).fetchall()]

        with patch.object(
            self.store,
            "get_relevant",
            side_effect=AssertionError("semantic retrieval must not run"),
        ), patch.object(
            self.store,
            "add",
            side_effect=AssertionError("memory-tool writes must not run"),
        ), patch.object(
            self.store.collection,
            "query",
            side_effect=AssertionError("Chroma must not be queried"),
        ), patch.object(
            self.store.collection,
            "add",
            side_effect=AssertionError("embeddings must not be created"),
        ):
            inspected = self.store.list_for_inspection()

        after = [tuple(row) for row in self.db.conn.execute(
            "SELECT id, category, content, importance, created_at, last_accessed_at FROM memory ORDER BY id"
        ).fetchall()]
        self.assertEqual(before, after)
        self.assertEqual([row["id"] for row in inspected], ["memory-c", "memory-b", "memory-a"])
        self.assertEqual(
            set(inspected[0]),
            {"id", "category", "content", "importance", "created_at", "last_accessed_at"},
        )
        self.assertEqual(len(inspected), 3)

    def test_inspection_empty_storage_returns_empty_list(self):
        self.assertEqual(self.store.list_for_inspection(), [])


if __name__ == "__main__":
    unittest.main()
