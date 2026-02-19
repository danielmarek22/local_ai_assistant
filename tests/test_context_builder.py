import unittest

from app.services.context_builder import ContextBuilder


class FakeHistoryStore:
    def __init__(self, rows):
        self.rows = rows
        self.last_limit = None

    def get_recent(self, session_id: str, limit: int):
        self.last_limit = limit
        return self.rows


class FakeMemoryStore:
    def __init__(self, memories):
        self.memories = memories
        self.last_query = None
        self.last_limit = None

    def get_relevant(self, query: str, limit: int):
        self.last_query = query
        self.last_limit = limit
        return self.memories


class FakeSummaryStore:
    def __init__(self, summary):
        self.summary = summary

    def get(self, session_id: str):
        return self.summary


class ContextBuilderTests(unittest.TestCase):
    def test_build_includes_system_tool_memory_history_and_user(self):
        history = FakeHistoryStore(
            [
                {"role": "assistant", "content": "Older assistant reply"},
                {"role": "user", "content": "First user msg"},
                {"role": "user", "content": "First user msg"},
                {"role": "user", "content": "Current question"},
            ]
        )
        memory = FakeMemoryStore(["User likes coffee"])
        summary = FakeSummaryStore("Conversation summary.")

        builder = ContextBuilder(
            system_prompt="System prompt",
            history_store=history,
            memory_store=memory,
            summary_store=summary,
            history_limit=6,
            memory_limit=5,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Current question",
            tool_context="External information:\nSome facts",
        )

        self.assertEqual(messages[0], {"role": "system", "content": "System prompt"})
        self.assertEqual(messages[1]["role"], "system")
        self.assertIn("External information:", messages[1]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Current question"})

        all_user_contents = [m["content"] for m in messages if m["role"] == "user"]
        self.assertEqual(all_user_contents.count("First user msg"), 1)
        self.assertNotIn("Older assistant reply", all_user_contents)

        self.assertEqual(history.last_limit, 2)
        self.assertEqual(memory.last_query, "Current question")
        self.assertEqual(memory.last_limit, 5)


if __name__ == "__main__":
    unittest.main()
