import tempfile
import unittest
import sqlite3
from pathlib import Path

from app.memory.chat_history import ChatHistoryStore
from app.perception.state import ImageAttachment
from app.storage.database import Database
from app.core.conversation import SenderAttribution, SenderType, InputSource, SessionKind
from app.services.turn_finalizer import TurnFinalizer


class FakeCollection:
    def __init__(self):
        self.records = []
        self.deleted_wheres = []

    def add(self, ids, documents, metadatas):
        for record_id, document, metadata in zip(ids, documents, metadatas):
            self.records.append(
                {
                    "id": record_id,
                    "document": document,
                    "metadata": metadata,
                }
            )

    def query(self, query_texts, n_results, where=None):
        records = self.records
        if where and "session_id" in where and "$ne" in where["session_id"]:
            excluded = where["session_id"]["$ne"]
            records = [
                record for record in records
                if record["metadata"].get("session_id") != excluded
            ]

        return {
            "documents": [[record["document"] for record in records[:n_results]]],
            "distances": [[record["metadata"].get("distance", 0.2) for record in records[:n_results]]],
        }

    def delete(self, where=None):
        self.deleted_wheres.append(where)
        if where and "session_id" in where:
            session_id = where["session_id"]
            self.records = [
                record for record in self.records
                if record["metadata"].get("session_id") != session_id
            ]


class FakeVectorStore:
    def __init__(self):
        self.semantic_collection = FakeCollection()
        self.episodic_collection = FakeCollection()


class FakeImageSummarizer:
    def __init__(self):
        self.calls = []

    def summarize(self, attachment, message_text: str = "") -> str:
        self.calls.append((attachment.name, message_text))
        return "Screenshot of the settings screen showing the speech toggle enabled."


class ChatHistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.db = Database(path=":memory:")
        self.vector_store = FakeVectorStore()
        self.image_summarizer = FakeImageSummarizer()
        self.store = ChatHistoryStore(
            self.db,
            self.vector_store,
            uploads_root=self.temp_dir.name,
            image_summarizer=self.image_summarizer,
        )

    def test_add_persists_image_summary_to_sqlite_and_vectordb(self):
        attachment = ImageAttachment(
            name="settings.png",
            mime_type="image/png",
            base64_data="aGVsbG8=",
            size_bytes=5,
        )

        message_id = self.store.add(
            "session-1",
            "user",
            "Please remember this screen",
            attachments=[attachment],
        )

        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT summary_text FROM chat_attachments WHERE message_id = ?",
            (message_id,),
        )
        row = cursor.fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(
            row["summary_text"],
            "Screenshot of the settings screen showing the speech toggle enabled.",
        )
        self.assertEqual(
            self.image_summarizer.calls,
            [("settings.png", "Please remember this screen")],
        )

        history_rows = self.store.get_all("session-1")
        self.assertEqual(len(history_rows), 1)
        self.assertEqual(len(history_rows[0]["attachments"]), 1)
        self.assertEqual(
            history_rows[0]["attachments"][0].summary_text,
            "Screenshot of the settings screen showing the speech toggle enabled.",
        )

        image_docs = [
            record for record in self.vector_store.episodic_collection.records
            if record["metadata"].get("source") == "image_attachment"
        ]
        self.assertEqual(len(image_docs), 1)
        self.assertIn(
            "Image summary: Screenshot of the settings screen showing the speech toggle enabled.",
            image_docs[0]["document"],
        )

        results = self.store.search_past_conversations(
            "speech toggle enabled",
            current_session="different-session",
            limit=5,
        )
        self.assertTrue(
            any("speech toggle enabled" in document for document in results)
        )

    def test_delete_session_removes_attachment_rows_files_and_vector_docs(self):
        attachment = ImageAttachment(
            name="settings.png",
            mime_type="image/png",
            base64_data="aGVsbG8=",
            size_bytes=5,
        )

        message_id = self.store.add(
            "session-1",
            "user",
            "Please remember this screen",
            attachments=[attachment],
        )
        attachment_dir = Path(self.temp_dir.name) / "session-1" / str(message_id)

        self.assertTrue(attachment_dir.exists())
        self.assertGreater(len(self.vector_store.episodic_collection.records), 0)

        deleted_count = self.store.delete_session("session-1")

        self.assertEqual(deleted_count, 1)
        self.assertEqual(self.store.get_all("session-1"), [])
        self.assertFalse(attachment_dir.exists())

        cursor = self.db.conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM chat_attachments WHERE session_id = ?", ("session-1",))
        self.assertEqual(cursor.fetchone()["count"], 0)
        cursor.execute("SELECT COUNT(*) AS count FROM chat_history WHERE session_id = ?", ("session-1",))
        self.assertEqual(cursor.fetchone()["count"], 0)

        self.assertEqual(
            self.vector_store.episodic_collection.deleted_wheres,
            [{"session_id": "session-1"}],
        )
        self.assertEqual(self.vector_store.episodic_collection.records, [])

    def test_additive_migration_and_legacy_sender_defaults(self):
        db_path = Path(self.temp_dir.name) / "legacy.db"
        connection = sqlite3.connect(db_path)
        connection.execute(
            """CREATE TABLE chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        connection.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES ('legacy', 'user', 'hello')"
        )
        connection.commit()
        connection.close()

        migrated = Database(str(db_path))
        columns = {row["name"] for row in migrated.conn.execute("PRAGMA table_info(chat_history)")}
        self.assertTrue({"sender_id", "sender_display_name", "sender_type", "input_source"} <= columns)
        store = ChatHistoryStore(
            migrated, self.vector_store, uploads_root=self.temp_dir.name,
            local_human_id="person-1", local_human_name="Local Person",
        )
        row = store.get_all("legacy")[0]
        self.assertEqual(
            (row["sender_id"], row["sender_display_name"], row["sender_type"], row["input_source"]),
            ("person-1", "Local Person", "human", "local_text"),
        )
        raw = migrated.conn.execute("SELECT sender_id FROM chat_history WHERE id = 1").fetchone()
        self.assertIsNone(raw["sender_id"])

    def test_failed_turn_is_retryable_and_resolves_without_duplicate_user_message(self):
        message_id = self.store.add("retry", "user", "Please answer this")

        attempts = self.store.mark_turn_failed(
            "retry", message_id, "Astra couldn't finish this response."
        )

        self.assertEqual(attempts, 1)
        retry_row = self.store.get_retryable_user_message("retry", message_id)
        self.assertEqual(retry_row["content"], "Please answer this")
        self.assertEqual(retry_row["retry_attempts"], 1)
        self.store.resolve_turn_failure("retry", message_id)
        self.assertIsNone(self.store.get_retryable_user_message("retry", message_id))
        self.assertEqual(self.store.count_messages("retry"), 1)

    def test_legacy_fallback_is_hidden_and_migrated_to_user_failure(self):
        db_path = Path(self.temp_dir.name) / "legacy-fallback.db"
        legacy = Database(str(db_path))
        legacy.conn.execute(
            """INSERT INTO chat_history (session_id, role, content)
               VALUES ('legacy-fallback', 'user', 'Try this')"""
        )
        legacy.conn.execute(
            """INSERT INTO chat_history (session_id, role, content)
               VALUES ('legacy-fallback', 'assistant', ?)""",
            ("I'm sorry, I lost my train of thought. Could you repeat that?",),
        )
        legacy.conn.commit()
        legacy.conn.close()

        migrated = Database(str(db_path))
        store = ChatHistoryStore(
            migrated, self.vector_store, uploads_root=self.temp_dir.name
        )

        rows = store.get_all("legacy-fallback")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["role"], "user")
        self.assertEqual(rows[0]["retry_attempts"], 1)
        self.assertIsNotNone(
            store.get_retryable_user_message("legacy-fallback", rows[0]["id"])
        )

    def test_group_session_and_vector_documents_preserve_sender(self):
        self.store.ensure_session("group-1", SessionKind.MANUAL_GROUP)
        sender = SenderAttribution(
            "relay:external_agent:abc", "Claude", SenderType.EXTERNAL_AGENT,
            InputSource.MANUAL_RELAY,
        )
        self.store.add("group-1", "user", "Same thought", sender=sender)

        self.assertEqual(self.store.get_session_kind("group-1"), SessionKind.MANUAL_GROUP)
        record = self.vector_store.episodic_collection.records[-1]
        self.assertEqual(record["document"], "EXTERNAL_AGENT Claude: Same thought")
        self.assertEqual(record["metadata"]["sender_id"], sender.sender_id)
        self.assertEqual(record["metadata"]["input_source"], "manual_relay")

        self.assertEqual(self.store.delete_session("group-1"), 1)
        self.assertEqual(self.store.get_session_kind("group-1"), SessionKind.DIRECT)

    def test_add_can_create_authoritative_group_session_on_first_message(self):
        self.store.add(
            "new-group",
            "user",
            "First group message",
            sender=SenderAttribution(
                "relay:human:alice", "Alice", SenderType.HUMAN,
                InputSource.MANUAL_RELAY,
            ),
            session_kind=SessionKind.MANUAL_GROUP,
        )

        self.assertEqual(
            self.store.get_session_kind("new-group"),
            SessionKind.MANUAL_GROUP,
        )
        self.assertEqual(
            self.vector_store.episodic_collection.records[-1]["document"],
            "HUMAN Alice: First group message",
        )

    def test_group_summary_input_preserves_participant_attribution(self):
        self.store.ensure_session("group-summary", SessionKind.MANUAL_GROUP)
        self.store.add(
            "group-summary", "user", "I am in Warsaw",
            sender=SenderAttribution(
                "relay:human:alice", "Alice", SenderType.HUMAN, InputSource.MANUAL_RELAY
            ),
        )

        class SummaryStore:
            def __init__(self): self.saved = None
            def get(self, _session_id): return None
            def set(self, session_id, summary, count): self.saved = (session_id, summary, count)

        class Summarizer:
            def __init__(self): self.messages = None
            def summarize(self, messages):
                self.messages = messages
                return "Alice said she is in Warsaw."

        summary_store = SummaryStore()
        summarizer = Summarizer()
        TurnFinalizer(self.store, summary_store, summarizer, summary_trigger=1).finalize("group-summary")

        self.assertIn('"sender_display_name":"Alice"', summarizer.messages[0]["content"])
        self.assertEqual(summary_store.saved, (
            "group-summary", "Alice said she is in Warsaw.", 1
        ))

    def test_group_participant_catalogue_uses_authoritative_history(self):
        self.store.ensure_session("group-catalogue", SessionKind.MANUAL_GROUP)
        alice = SenderAttribution(
            "relay:human:alice", "Alice", SenderType.HUMAN,
            InputSource.MANUAL_RELAY,
        )
        chatgpt = SenderAttribution(
            "relay:external_agent:chatgpt", "ChatGPT", SenderType.EXTERNAL_AGENT,
            InputSource.MANUAL_RELAY,
        )
        first_alice_id = self.store.add(
            "group-catalogue", "user", "I prefere green tea", sender=alice,
        )
        self.store.add("group-catalogue", "assistant", "Acknowledged")
        self.store.add(
            "group-catalogue", "user", "I prefere espresso", sender=chatgpt,
        )
        latest_alice_id = self.store.add(
            "group-catalogue", "user", "Still here", sender=SenderAttribution(
                alice.sender_id, "Alice Cooper", SenderType.HUMAN,
                InputSource.MANUAL_RELAY,
            ),
        )
        current_id = self.store.add(
            "group-catalogue", "user", "Alice prefers espresso", sender=chatgpt,
        )

        participants = self.store.get_participant_senders_before(
            "group-catalogue", current_id,
        )

        self.assertEqual(
            [item["sender_id"] for item in participants],
            [alice.sender_id, chatgpt.sender_id],
        )
        self.assertEqual(participants[0]["sender_display_name"], "Alice Cooper")
        self.assertEqual(participants[0]["latest_id"], latest_alice_id)
        self.assertNotEqual(participants[0]["latest_id"], first_alice_id)
        self.assertEqual(
            len(self.store.get_participant_senders_before(
                "group-catalogue", current_id, limit=1,
            )),
            1,
        )


if __name__ == "__main__":
    unittest.main()
