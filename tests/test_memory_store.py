import unittest

from app.storage.database import Database
from app.memory.memory_store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(path=":memory:")
        self.store = MemoryStore(self.db)

    def test_get_relevant_prefers_overlap_then_importance(self):
        self.store.add("I like chess", importance=1)
        self.store.add("My favorite color is blue", importance=3)
        self.store.add("I enjoy chess openings", importance=1)

        results = self.store.get_relevant("chess openings", limit=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "I enjoy chess openings")
        self.assertIn("I like chess", results)

    def test_high_importance_without_overlap_is_kept(self):
        self.store.add("Unrelated but important memory", importance=3)
        self.store.add("Another unrelated low-priority memory", importance=1)

        results = self.store.get_relevant("chess", limit=5)

        self.assertIn("Unrelated but important memory", results)
        self.assertNotIn("Another unrelated low-priority memory", results)


if __name__ == "__main__":
    unittest.main()
