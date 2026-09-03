import base64
import hashlib
import unittest
from datetime import datetime

from app.perception.attachments import AudioAttachment
from app.perception.state import ImageAttachment
from app.services.context_builder import ContextBuilder
from app.core.conversation import SenderAttribution, SenderType, InputSource, SessionKind


class FakeHistoryStore:
    def __init__(self, rows):
        self.rows = rows
        self.last_limit = None

    def get_recent(self, session_id: str, limit: int):
        self.last_limit = limit
        return self.rows

    def count_messages(self, session_id: str):
        return len(self.rows)


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
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Current question",
            memory_context="Some retrieved facts from VectorDB",
            integration_context="Some integration state",
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
        self.assertIn("BACKGROUND CONTEXT (Retrieved Memories & Integration State):", system_content)
        self.assertIn("--- RETRIEVED MEMORY ---", system_content)
        self.assertIn("Some retrieved facts from VectorDB", system_content)
        self.assertIn("--- OBSERVED INTEGRATION STATE ---", system_content)
        self.assertIn("Some integration state", system_content)
        
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

    def test_build_attaches_audio_to_configured_payload_field(self):
        history = FakeHistoryStore([])
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            history_store=history,
            summary_store=summary,
            history_limit=6,
            audio_payload_field="audios",
        )

        attachment = AudioAttachment.from_bytes(
            b"audio-bytes",
            name="voice.wav",
            mime_type="audio/wav",
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Please answer the user's spoken audio.",
            attachments=[attachment],
        )

        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["audios"], [attachment.base64_data])

    def test_build_prepends_audio_when_using_images_payload_field(self):
        history = FakeHistoryStore([])
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            history_store=history,
            summary_store=summary,
            history_limit=6,
            audio_payload_field="images",
        )

        audio = AudioAttachment.from_bytes(b"audio-bytes")
        image = ImageAttachment(
            name="cat.png",
            mime_type="image/png",
            base64_data="aW1hZ2U=",
            size_bytes=5,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Please answer with both inputs.",
            attachments=[image, audio],
        )

        self.assertEqual(messages[-1]["images"], [audio.base64_data, "aW1hZ2U="])

    def test_build_replaces_recent_history_images_with_text_context(self):
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
                            summary_text="A cat sitting on a desk.",
                        )
                    ],
                }
            ]
        )
        summary = FakeSummaryStore(None)

        builder = ContextBuilder(
            system_prompt="System prompt",
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        messages = builder.build(
            session_id="abc123",
            user_text="Can you compare it?",
        )

        self.assertEqual(messages[-2]["role"], "user")
        self.assertEqual(
            messages[-2]["content"],
            "Earlier screenshot\n"
            "[Earlier attached image: earlier.png. Image summary: A cat sitting on a desk.]",
        )
        self.assertNotIn("images", messages[-2])

    def test_build_mentions_unsummarized_history_image_without_replaying_it(self):
        history = FakeHistoryStore([
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
        ])
        builder = ContextBuilder(
            "System prompt", history, summary_store=FakeSummaryStore(None)
        )

        messages = builder.build("abc123", "What did I send?")

        self.assertEqual(
            messages[-2]["content"],
            "Earlier screenshot\n[Earlier attached image: earlier.png]",
        )
        self.assertNotIn("images", messages[-2])

    def test_build_unwraps_summary_store_tuple(self):
        history = FakeHistoryStore([])
        summary = FakeSummaryStore(("Conversation summary.", 4))

        builder = ContextBuilder(
            system_prompt="System prompt",
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

    def test_build_keeps_every_message_since_summary_checkpoint(self):
        history = FakeHistoryStore([
            {"role": "user", "content": f"Message {index}"}
            for index in range(9)
        ])
        builder = ContextBuilder(
            system_prompt="System prompt",
            history_store=history,
            summary_store=FakeSummaryStore(("Summary through message 4.", 4)),
            history_limit=10,
        )

        builder.build(session_id="abc123", user_text="Current question")

        self.assertEqual(history.last_limit, 5)

    def test_build_deduplicates_current_user_message_against_stored_attachment_variant(self):
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        image_sha256 = hashlib.sha256(base64.b64decode(image_b64)).hexdigest()
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
                            "size_bytes": 69,
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
            history_store=history,
            summary_store=summary,
            history_limit=6,
        )

        attachment = ImageAttachment.from_payload(
            {
                "name": "cat.png",
                "mime_type": "image/png",
                "data": image_b64,
                "size_bytes": 69,
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

    def test_group_context_attributes_identical_text_from_different_senders(self):
        history = FakeHistoryStore([
            {
                "role": "user", "content": "I agree", "attachments": [],
                "sender_id": "relay:human:a", "sender_display_name": "Alice",
                "sender_type": "human", "input_source": "manual_relay",
            }
        ])
        builder = ContextBuilder("System prompt", history, summary_store=FakeSummaryStore(None))
        bob = SenderAttribution(
            "relay:human:b", "Bob", SenderType.HUMAN, InputSource.MANUAL_RELAY
        )

        messages = builder.build(
            "group", "I agree", current_sender=bob, session_kind=SessionKind.MANUAL_GROUP
        )

        user_messages = [message for message in messages if message["role"] == "user"]
        self.assertEqual(len(user_messages), 2)
        self.assertIn('"sender_display_name":"Alice"', user_messages[0]["content"])
        self.assertIn('"sender_display_name":"Bob"', user_messages[1]["content"])
        self.assertIn("MANUAL GROUP CHAT ATTRIBUTION", messages[0]["content"])

    def test_direct_context_format_remains_plain(self):
        builder = ContextBuilder("System prompt", FakeHistoryStore([]), summary_store=FakeSummaryStore(None))
        messages = builder.build("direct", "Hello", session_kind=SessionKind.DIRECT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Hello"})
        self.assertNotIn("MANUAL GROUP CHAT ATTRIBUTION", messages[0]["content"])

    def test_group_context_without_authoritative_sender_omits_synthetic_participant(self):
        builder = ContextBuilder(
            "System prompt",
            FakeHistoryStore([]),
            summary_store=FakeSummaryStore(None),
        )
        messages = builder.build(
            "group",
            "",
            session_kind=SessionKind.MANUAL_GROUP,
        )

        self.assertEqual([message["role"] for message in messages], ["system"])
        self.assertNotIn('"sender_display_name":"You"', messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
