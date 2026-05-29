import tempfile
import unittest
from pathlib import Path

from app.memory.chat_history import ChatHistoryStore
from app.perception.state import ImageAttachment
from app.storage.database import Database


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


if __name__ == "__main__":
    unittest.main()
