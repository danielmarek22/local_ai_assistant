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
    def test_build_includes_system_tool_memory_conversation_and_user(self):
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
            user_context={},
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
        system_contents = [m["content"] for m in messages if m["role"] == "system"]
        datetime_line = next(
            (c for c in system_contents if c.startswith("Current system datetime (local): ")),
            None,
        )
        self.assertIsNotNone(datetime_line)
        self.assertIsNotNone(
            datetime.fromisoformat(datetime_line.split(": ", 1)[1]),
        )
        self.assertTrue(
            any("External information:" in c for c in system_contents),
        )
        self.assertEqual(messages[-1], {"role": "user", "content": "Current question"})

        all_user_contents = [m["content"] for m in messages if m["role"] == "user"]
        all_assistant_contents = [
            m["content"] for m in messages if m["role"] == "assistant"
        ]
        self.assertEqual(all_user_contents.count("First user msg"), 1)
        self.assertIn("Older assistant reply", all_assistant_contents)

        self.assertEqual(history.last_limit, 2)
        self.assertEqual(memory.last_query, "Current question")
        self.assertEqual(memory.last_limit, 5)

    def test_build_includes_configured_user_context(self):
        history = FakeHistoryStore([])
        memory = FakeMemoryStore([])
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            user_context={
                "name": "Bob",
                "timezone": "America/New_York",
                "preferences": "concise answers",
            },
            history_store=history,
            memory_store=memory,
            summary_store=summary,
            history_limit=6,
            memory_limit=5,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Hello",
            tool_context=None,
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
        self.assertIn("User profile/context (configured):", user_context_message)
        self.assertIn("- name: Bob", user_context_message)
        self.assertIn("- timezone: America/New_York", user_context_message)
        self.assertIn("- preferences: concise answers", user_context_message)

    def test_build_includes_multiline_user_profile_inside_user_context(self):
        history = FakeHistoryStore([])
        memory = FakeMemoryStore([])
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            user_context={
                "profile": (
                    "User has ADHD.\n"
                    "Prefers short, direct responses."
                ),
            },
            history_store=history,
            memory_store=memory,
            summary_store=summary,
            history_limit=6,
            memory_limit=5,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Hello",
            tool_context=None,
        )

        profile_message = next(
            (
                m["content"]
                for m in messages
                if m["role"] == "system"
                and "- profile:" in m["content"]
            ),
            "",
        )
        self.assertIn(
            "- profile:",
            profile_message,
        )
        self.assertIn(
            "User has ADHD.",
            profile_message,
        )
        self.assertIn(
            "Prefers short, direct responses.",
            profile_message,
        )


if __name__ == "__main__":
    unittest.main()
