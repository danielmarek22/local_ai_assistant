import unittest
import json
from unittest.mock import Mock, patch

from app.memory.reflector import MemoryReflector
from app.llm.ollama_stream import OllamaClient


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.last_options_override = None
        self.last_think_override = None

    def chat(self, messages, think_override=None, options_override=None):
        self.calls += 1
        self.last_think_override = think_override
        self.last_options_override = options_override
        return {"role": "assistant", "content": self.response}


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

    def apply_consolidation(self, delete_ids, new_memories):
        self.deleted_ids.extend(delete_ids)
        self.added_memories.extend(new_memories)
        return {"deleted_count": len(delete_ids), "index_sync_complete": True, "index_sync_errors": []}

    def add(self, content, category="general", importance=1):
        self.added_memories.append(
            {
                "content": content,
                "category": category,
                "importance": importance,
            }
        )


class MemoryReflectorTests(unittest.TestCase):
    def test_committed_reflection_reports_index_repair_without_claiming_rollback(self):
        store = FakeMemoryStore([{"id": "old", "importance": 2, "content": "Old"}])
        store.apply_consolidation = Mock(return_value={
            "deleted_count": 1, "index_sync_complete": False,
            "index_sync_errors": ["Replacement memories could not be indexed"],
        })
        llm = FakeLLM(json.dumps({"delete_ids": ["old", "old"], "keep_ids": [],
                                 "new_memories": [{"content": "Replacement"}]}))
        result = MemoryReflector(llm, store).reflect_and_prune(0)
        self.assertTrue(result["success"])
        self.assertEqual(result["deleted_count"], 1)
        self.assertEqual(result["created_count"], 1)
        self.assertFalse(result["index_sync_complete"])
        self.assertEqual(len(result["index_sync_errors"]), 1)
        self.assertEqual(store.apply_consolidation.call_args.args[0], ["old"])

    def test_real_chat_adapter_message_is_used_without_parsing_thinking(self):
        llm = OllamaClient(model="test", host="http://unused")
        store = FakeMemoryStore([{"id": "old", "importance": 2, "content": "Old"}])
        response = Mock()
        response.json.return_value = {"message": {
            "role": "assistant", "thinking": '{"delete_ids": ["old"]}',
            "content": json.dumps({"delete_ids": [], "keep_ids": ["old"], "new_memories": []}),
        }}
        with patch.object(llm, "_post_with_retry", return_value=response):
            result = MemoryReflector(llm, store).reflect_and_prune(0)
        self.assertTrue(result["success"])
        self.assertEqual(result["keep_ids"], ["old"])
        self.assertEqual(store.deleted_ids, [])

    def test_bad_response_shapes_and_invalid_plans_never_mutate_memory(self):
        for response in (None, "raw text", [], {}, {"content": None}, {"content": []},
                         {"content": "  "}, {"content": '{"delete_ids": ['},
                         {"content": '{"delete_ids": ["old"], "keep_ids": []}'},
                         {"content": '{"delete_ids": ["old"], "keep_ids": ["old"], "new_memories": []}'}):
            with self.subTest(response=response):
                store = FakeMemoryStore([{"id": "old", "importance": 2, "content": "Old"}])
                llm = Mock()
                llm.chat.return_value = response
                with self.assertLogs("memory_reflector", level="ERROR"):
                    result = MemoryReflector(llm, store).reflect_and_prune(0)
                self.assertFalse(result["success"])
                self.assertEqual(store.deleted_ids, [])
                self.assertEqual(store.added_memories, [])

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
        self.assertEqual(llm.last_think_override, False)
        self.assertEqual(llm.last_options_override, {"temperature": 0.0, "num_predict": 1024})

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
