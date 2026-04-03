import unittest

from app.services.memory_reflector import MemoryReflector


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def chat(self, messages, _think_override):
        self.calls += 1
        return self.response


class FakeMemoryStore:
    def __init__(self, stale_memories):
        self.stale_memories = stale_memories
        self.deleted_ids = []
        self.added_memories = []

    def get_stale(self, days_old):
        self.last_days_old = days_old
        return self.stale_memories

    def delete_memories(self, memory_ids):
        self.deleted_ids.extend(memory_ids)

    def add(self, content, category="general", importance=1):
        self.added_memories.append(
            {
                "content": content,
                "category": category,
                "importance": importance,
            }
        )


class MemoryReflectorTests(unittest.TestCase):
    def test_reflect_and_prune_applies_deletes_and_additions(self):
        stale = [
            {"id": "mem-1", "importance": 2, "content": "Specific old fact"},
            {"id": "mem-2", "importance": 3, "content": "Another old fact"},
        ]
        llm = FakeLLM(
            """
            {
              "delete_ids": ["mem-1"],
              "keep_ids": ["mem-2"],
              "new_memories": [
                {"content": "User prefers concise Python.", "category": "preference", "importance": 3}
              ]
            }
            """
        )
        memory_store = FakeMemoryStore(stale)
        reflector = MemoryReflector(llm=llm, memory_store=memory_store)

        result = reflector.reflect_and_prune(days_old=0)

        self.assertTrue(result["success"])
        self.assertEqual(result["stale_count"], 2)
        self.assertEqual(result["deleted_count"], 1)
        self.assertEqual(result["kept_count"], 1)
        self.assertEqual(result["created_count"], 1)
        self.assertEqual(memory_store.deleted_ids, ["mem-1"])
        self.assertEqual(len(memory_store.added_memories), 1)
        self.assertEqual(memory_store.added_memories[0]["category"], "preference")

    def test_reflect_and_prune_returns_zero_op_when_no_stale_memories(self):
        llm = FakeLLM("should not be called")
        memory_store = FakeMemoryStore(stale_memories=[])
        reflector = MemoryReflector(llm=llm, memory_store=memory_store)

        result = reflector.reflect_and_prune(days_old=14)

        self.assertTrue(result["success"])
        self.assertEqual(result["stale_count"], 0)
        self.assertEqual(result["deleted_count"], 0)
        self.assertEqual(result["created_count"], 0)
        self.assertEqual(llm.calls, 0)

    def test_reflect_and_prune_returns_failure_for_invalid_llm_json(self):
        stale = [{"id": "mem-1", "importance": 2, "content": "Old fact"}]
        llm = FakeLLM("not valid json")
        memory_store = FakeMemoryStore(stale_memories=stale)
        reflector = MemoryReflector(llm=llm, memory_store=memory_store)

        result = reflector.reflect_and_prune(days_old=7)

        self.assertFalse(result["success"])
        self.assertEqual(result["stale_count"], 1)
        self.assertIn("No JSON object found", result["error"])

    def test_reflect_and_prune_ignores_non_stale_delete_ids(self):
        stale = [{"id": "mem-1", "importance": 2, "content": "Old fact"}]
        llm = FakeLLM(
            """
            {
              "delete_ids": ["mem-1", "mem-999"],
              "keep_ids": [],
              "new_memories": []
            }
            """
        )
        memory_store = FakeMemoryStore(stale_memories=stale)
        reflector = MemoryReflector(llm=llm, memory_store=memory_store)

        result = reflector.reflect_and_prune(days_old=7)

        self.assertTrue(result["success"])
        self.assertEqual(result["delete_ids"], ["mem-1"])
        self.assertEqual(result["deleted_count"], 1)
        self.assertEqual(result["ignored_delete_ids"], ["mem-999"])
        self.assertEqual(memory_store.deleted_ids, ["mem-1"])


if __name__ == "__main__":
    unittest.main()
