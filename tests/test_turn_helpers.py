import unittest

from app.core.actions import Action, ActionType
from app.services.memory_action_handler import MemoryActionHandler
from app.services.memory_retriever import MemoryRetriever


class FakeMemoryStore:
    def __init__(self, relevant=None):
        self.relevant = relevant or []
        self.writes = []

    def get_relevant(self, query: str, limit: int = 3):
        return self.relevant[:limit]

    def add(self, content: str, category: str = "general", importance: int = 1):
        self.writes.append((content, category, importance))


class FakeHistoryStore:
    def __init__(self, episodic=None):
        self.episodic = episodic or []

    def search_past_conversations(self, query: str, current_session: str, limit: int = 4):
        return self.episodic[:limit]


class FakeDecision:
    def __init__(self, content, category="general", importance=2):
        self.content = content
        self.category = category
        self.importance = importance


class FakeMemoryPolicy:
    def __init__(self, decision=None):
        self.decision = decision

    def decide_from_action(self, action_payload: dict):
        return self.decision


class MemoryRetrieverTests(unittest.TestCase):
    def test_retrieve_builds_combined_context_and_perception_value(self):
        retriever = MemoryRetriever(
            memory_store=FakeMemoryStore(relevant=["User likes testing"]),
            history_store=FakeHistoryStore(episodic=["USER: Past question", "ASSISTANT: Past answer"]),
        )

        result = retriever.retrieve("hello", "session-1")

        self.assertIn("Relevant Facts:", result.memory_context)
        self.assertIn("User likes testing", result.memory_context)
        self.assertIn("Past Conversations:", result.memory_context)
        self.assertIn("Past answer", result.memory_context)
        self.assertEqual(result.perception_value, f"\n{result.memory_context}\n")

    def test_retrieve_returns_empty_fallback_when_query_or_results_missing(self):
        retriever = MemoryRetriever(
            memory_store=FakeMemoryStore(),
            history_store=FakeHistoryStore(),
        )

        empty_query = retriever.retrieve("", "session-1")
        no_results = retriever.retrieve("hello", "session-1")

        self.assertIsNone(empty_query.memory_context)
        self.assertEqual(empty_query.perception_value, "No relevant past memories found.")
        self.assertIsNone(no_results.memory_context)
        self.assertEqual(no_results.perception_value, "No relevant past memories found.")


class MemoryActionHandlerTests(unittest.TestCase):
    def test_handle_writes_memory_when_policy_returns_decision(self):
        memory = FakeMemoryStore()
        handler = MemoryActionHandler(
            memory_store=memory,
            memory_policy=FakeMemoryPolicy(
                decision=FakeDecision("remember this", category="prefs", importance=3)
            ),
        )

        handler.handle(
            "session-1",
            Action(type=ActionType.WRITE_MEMORY, payload={"content": "remember this"}),
        )

        self.assertEqual(memory.writes, [("remember this", "prefs", 3)])

    def test_handle_ignores_action_when_policy_returns_none(self):
        memory = FakeMemoryStore()
        handler = MemoryActionHandler(
            memory_store=memory,
            memory_policy=FakeMemoryPolicy(decision=None),
        )

        handler.handle(
            "session-1",
            Action(type=ActionType.WRITE_MEMORY, payload={"content": "remember this"}),
        )

        self.assertEqual(memory.writes, [])


if __name__ == "__main__":
    unittest.main()
