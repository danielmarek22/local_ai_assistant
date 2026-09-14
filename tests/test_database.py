import sqlite3
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from app.storage.database import Database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(path=":memory:")
        self.addCleanup(self.db.close)

    def test_rejects_attachment_without_parent_message(self):
        self.assertEqual(
            self.db.conn.execute("PRAGMA foreign_keys").fetchone()[0],
            1,
        )

        with self.assertRaises(sqlite3.IntegrityError):
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO chat_attachments (
                        message_id, session_id, name, mime_type, storage_path, sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (999, "session-1", "orphan.png", "image/png", "orphan.png", "digest"),
                )

    def test_deleting_message_cascades_to_attachments(self):
        with self.db.transaction() as conn:
            message_id = conn.execute(
                """
                INSERT INTO chat_history (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                ("session-1", "user", "Has attachment"),
            ).lastrowid
            conn.execute(
                """
                INSERT INTO chat_attachments (
                    message_id, session_id, name, mime_type, storage_path, sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, "session-1", "image.png", "image/png", "image.png", "digest"),
            )

        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chat_history WHERE id = ?", (message_id,))

        with self.db.connection() as conn:
            attachment_count = conn.execute(
                "SELECT COUNT(*) FROM chat_attachments WHERE message_id = ?",
                (message_id,),
            ).fetchone()[0]
        self.assertEqual(attachment_count, 0)

    def test_transactions_serialize_threads_and_isolate_rollback(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        second_attempted = threading.Event()
        second_entered = threading.Event()

        def commit_first():
            with self.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO memory (id, content) VALUES (?, ?)",
                    ("first", "committed"),
                )
                first_entered.set()
                if not release_first.wait(timeout=2.0):
                    raise TimeoutError("test did not release first transaction")

        def rollback_second():
            second_attempted.set()
            with self.db.transaction() as conn:
                second_entered.set()
                conn.execute(
                    "INSERT INTO memory (id, content) VALUES (?, ?)",
                    ("second", "rolled back"),
                )
                raise RuntimeError("force rollback")

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(commit_first)
            self.assertTrue(first_entered.wait(timeout=1.0))
            second_future = executor.submit(rollback_second)
            self.assertTrue(second_attempted.wait(timeout=1.0))
            try:
                self.assertFalse(second_entered.wait(timeout=0.1))
            finally:
                release_first.set()

            first_future.result(timeout=1.0)
            with self.assertRaisesRegex(RuntimeError, "force rollback"):
                second_future.result(timeout=1.0)

        with self.db.connection() as conn:
            rows = conn.execute("SELECT id FROM memory ORDER BY id").fetchall()
        self.assertEqual([row["id"] for row in rows], ["first"])


if __name__ == "__main__":
    unittest.main()
