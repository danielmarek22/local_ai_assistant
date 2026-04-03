import unittest
from app.storage.database import Database
from app.memory.memory_store import MemoryStore


class FakeCollection:
    def __init__(self):
        self.ids = []
        self.docs = []

    def add(self, ids, documents, metadatas):
        self.ids.extend(ids)
        self.docs.extend(documents)

    def query(self, query_texts, n_results, where=None):
        # Fake search behavior: return everything we have up to n_results
        return {
            "ids": [self.ids[:n_results]] if self.ids else [[]],
            "documents": [self.docs[:n_results]] if self.docs else [[]],
        }


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

        # Our fake collection just returns what it has
        results = self.store.get_relevant("find A", limit=2)
        
        self.assertEqual(len(results), 2)
        self.assertIn("Memory A", results)


if __name__ == "__main__":
    unittest.main()
