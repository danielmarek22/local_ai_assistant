import unittest
import hashlib
from datetime import datetime

from app.perception.state import ImageAttachment
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
    def test_build_includes_system_history_and_background_context(self):
        history = FakeHistoryStore(
            [
                {"role": "assistant", "content": "Older assistant reply"},
                {"role": "user", "content": "First user msg"},
                {"role": "user", "content": "Current question"},
            ]
        )
        summary = FakeSummaryStore("Conversation summary.")

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
            memory_context="Some retrieved facts from VectorDB",
            tool_context="Some tool results",
        )

        system_messages = [m for m in messages if m["role"] == "system"]
        self.assertEqual(len(system_messages), 1)

        system_content = system_messages[0]["content"]
        self.assertTrue(system_content.startswith("System prompt"))
        self.assertIn("\n\n---\n\n", system_content)

        datetime_line = next(
            (
                line
                for line in system_content.splitlines()
                if line.startswith("Current system datetime (local): ")
            ),
            None,
        )
        self.assertIsNotNone(datetime_line)
        self.assertIsNotNone(datetime.fromisoformat(datetime_line.split(": ", 1)[1]))

        # Verify the new structured background context
        self.assertIn("BACKGROUND CONTEXT (Retrieved Memories & Tool Results):", system_content)
        self.assertIn("--- RETRIEVED MEMORY ---", system_content)
        self.assertIn("Some retrieved facts from VectorDB", system_content)
        self.assertIn("--- TOOL RESULTS ---", system_content)
        self.assertIn("Some tool results", system_content)
        
        self.assertIn("Summary of previous conversation:\nConversation summary.", system_content)

        self.assertEqual(messages[-1], {"role": "user", "content": "Current question"})

        # Check history deduplication and inclusion
        all_user_contents = [m["content"] for m in messages if m["role"] == "user"]
        all_assistant_contents = [m["content"] for m in messages if m["role"] == "assistant"]
        self.assertEqual(all_user_contents.count("First user msg"), 1)
        self.assertIn("Older assistant reply", all_assistant_contents)

    def test_build_attaches_images_to_current_user_message(self):
        history = FakeHistoryStore([])
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            user_context={},
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        attachment = ImageAttachment(
            name="cat.png",
            mime_type="image/png",
            base64_data="aGVsbG8=",
            size_bytes=5,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="What is in this image?",
            attachments=[attachment],
        )

        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "What is in this image?")
        self.assertEqual(messages[-1]["images"], ["aGVsbG8="])

    def test_build_replays_recent_history_images_for_user_messages(self):
        history = FakeHistoryStore(
            [
                {
                    "role": "user",
                    "content": "Earlier screenshot",
                    "attachments": [
                        ImageAttachment(
                            name="earlier.png",
                            mime_type="image/png",
                            base64_data="aGVsbG8=",
                            size_bytes=5,
                        )
                    ],
                }
            ]
        )
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            user_context={},
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Can you compare it?",
        )

        self.assertEqual(messages[-2]["role"], "user")
        self.assertEqual(messages[-2]["content"], "Earlier screenshot")
        self.assertEqual(messages[-2]["images"], ["aGVsbG8="])

    def test_build_includes_configured_user_context(self):
        history = FakeHistoryStore([])
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            user_context={
                "name": "Bob",
                "preferences": "concise answers",
            },
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Hello",
            # injected_context removed from here completely
        )

        system_messages = [m for m in messages if m["role"] == "system"]
        self.assertEqual(len(system_messages), 1)

        system_content = system_messages[0]["content"]
        self.assertIn("User profile/context (configured):", system_content)
        self.assertIn("- name: Bob", system_content)
        self.assertIn("- preferences: concise answers", system_content)

    def test_build_unwraps_summary_store_tuple(self):
        history = FakeHistoryStore([])
        summary = FakeSummaryStore(("Conversation summary.", 4))

        builder = ContextBuilder(
            system_prompt="System prompt",
            user_context={},
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Hello",
        )

        system_content = messages[0]["content"]
        self.assertIn("Summary of previous conversation:\nConversation summary.", system_content)
        self.assertNotIn("('Conversation summary.', 4)", system_content)

    def test_build_deduplicates_current_user_message_against_stored_attachment_variant(self):
        image_b64 = "aGVsbG8="
        image_sha256 = hashlib.sha256(b"hello").hexdigest()
        history = FakeHistoryStore(
            [
                {
                    "role": "user",
                    "content": "Current question",
                    "attachments": [
                        {
                            "id": 42,
                            "name": "cat.png",
                            "mime_type": "image/png",
                            "size_bytes": 5,
                            "storage_path": "static/uploads/s1/1/cat.png",
                            "url": "/static/uploads/s1/1/cat.png",
                            "sha256": image_sha256,
                        }
                    ],
                }
            ]
        )
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            user_context={},
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        attachment = ImageAttachment.from_payload(
            {
                "name": "cat.png",
                "mime_type": "image/png",
                "data": image_b64,
                "size_bytes": 5,
            }
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Current question",
            attachments=[attachment],
        )

        user_messages = [message for message in messages if message["role"] == "user"]
        self.assertEqual(len(user_messages), 1)
        self.assertEqual(user_messages[0]["content"], "Current question")
        self.assertEqual(user_messages[0]["images"], [image_b64])


if __name__ == "__main__":
    unittest.main()
