import unittest
from datetime import datetime

from app.services.context_builder import ContextBuilder


class FakeHistoryStore:
    def __init__(self, rows):
        self.rows = rows
        self.last_limit = None

    def get_recent(self, session_id: str, limit: int):
        self.last_limit = limit
        return self.rows


class FakeSummaryStore:
    def __init__(self, summary):
        self.summary = summary

    def get(self, session_id: str):
        return self.summary


class ContextBuilderTests(unittest.TestCase):
    def test_build_includes_system_history_and_injected_context(self):
        history = FakeHistoryStore(
            [
                {"role": "assistant", "content": "Older assistant reply"},
                {"role": "user", "content": "First user msg"},
                {"role": "user", "content": "Current question"},
            ]
        )
        summary = FakeSummaryStore("Conversation summary.")

        # memory_store and memory_limit removed from instantiation
        builder = ContextBuilder(
            system_prompt="System prompt",
            user_context={},
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Current question",
            injected_context="Some retrieved facts from VectorDB",
        )

        self.assertEqual(messages[0], {"role": "system", "content": "System prompt"})
        system_contents = [m["content"] for m in messages if m["role"] == "system"]
        
        # Check datetime
        datetime_line = next(
            (c for c in system_contents if c.startswith("Current system datetime (local): ")),
            None,
        )
        self.assertIsNotNone(datetime_line)
        self.assertIsNotNone(datetime.fromisoformat(datetime_line.split(": ", 1)[1]))
        
        # Check injected context format
        self.assertTrue(
            any("BACKGROUND CONTEXT (Retrieved Memories & Tool Results):" in c for c in system_contents),
        )
        self.assertTrue(any("Some retrieved facts from VectorDB" in c for c in system_contents))

        self.assertEqual(messages[-1], {"role": "user", "content": "Current question"})

        # Check history deduplication and inclusion
        all_user_contents = [m["content"] for m in messages if m["role"] == "user"]
        all_assistant_contents = [m["content"] for m in messages if m["role"] == "assistant"]
        self.assertEqual(all_user_contents.count("First user msg"), 1)
        self.assertIn("Older assistant reply", all_assistant_contents)

    def test_build_includes_configured_user_context(self):
        history = FakeHistoryStore([])
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            user_context={
                "name": "Bob",
                "timezone": "America/New_York",
                "preferences": "concise answers",
            },
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Hello",
            injected_context=None,
        )

        user_context_message = next(
            (
                m["content"]
                for m in messages
                if m["role"] == "system"
                and "User profile/context (configured):" in m["content"]
            ),
            "",
        )
        self.assertIn("- name: Bob", user_context_message)
        self.assertIn("- preferences: concise answers", user_context_message)


if __name__ == "__main__":
    unittest.main()