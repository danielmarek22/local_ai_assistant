import unittest
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app.core.turn_finalizer import TurnFinalizer
from app.memory.chat_history import ChatHistoryStore
from app.memory.summary_store import SummaryStore
from app.storage.database import Database


class SummaryCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.close)
        self.history = ChatHistoryStore(self.db, SimpleNamespace(episodic_collection=Mock()))
        self.store = SummaryStore(self.db)
        self.summarizer = Mock()
        self.summarizer.summarize.return_value = "Updated summary"
        self.finalizer = TurnFinalizer(self.history, self.store, self.summarizer, summary_trigger=2)

    def seed(self, count, *, session="session", role="user", excluded=0):
        with self.db.transaction() as conn:
            for number in range(count):
                conn.execute(
                    "INSERT INTO chat_history (session_id, role, content, excluded_from_context) VALUES (?, ?, ?, ?)",
                    (session, role, f"{session}-{role}-{number}", excluded),
                )
            return conn.execute("SELECT MAX(id) FROM chat_history").fetchone()[0]

    def test_checkpoint_at_one_thousand_does_not_stall(self):
        checkpoint = self.seed(1000)
        self.store.set("session", "Previous summary", checkpoint)
        last_id = self.seed(2)
        self.finalizer.finalize("session")
        self.summarizer.summarize.assert_called_once()
        self.assertEqual(len(self.summarizer.summarize.call_args.args[0]), 2)
        self.assertEqual(self.store.get("session"), ("Updated summary", last_id))

    def test_backlog_is_processed_in_order_in_bounded_batches(self):
        last_id = self.seed(1100)
        for _ in range(11):
            self.finalizer.finalize("session")
        batches = [call.args[0] for call in self.summarizer.summarize.call_args_list]
        self.assertEqual([len(batch) for batch in batches], [100] * 11)
        self.assertEqual([row["content"] for batch in batches for row in batch],
                         [f"session-user-{number}" for number in range(1100)])
        self.assertEqual(self.store.get("session")[1], last_id)
        self.finalizer.finalize("session")
        self.assertEqual(self.summarizer.summarize.call_count, 11)

    def test_tools_exclusions_and_other_sessions_do_not_count_or_shift_checkpoint(self):
        self.seed(1)
        self.seed(150, role="tool")
        self.seed(3, excluded=1)
        self.seed(3, session="other")
        self.finalizer.finalize("session")
        self.summarizer.summarize.assert_not_called()
        last_id = self.seed(1, role="assistant")
        self.finalizer.finalize("session")
        messages = self.summarizer.summarize.call_args.args[0]
        self.assertEqual([row["role"] for row in messages], ["user", "assistant"])
        self.assertEqual(self.store.get("session")[1], last_id)
        with self.db.transaction() as conn:
            conn.execute("UPDATE chat_history SET excluded_from_context = 1 WHERE id = 1")
        last_id = self.seed(2)
        self.finalizer.finalize("session")
        self.assertEqual(len(self.summarizer.summarize.call_args.args[0]), 2)
        self.assertEqual(self.store.get("session")[1], last_id)

    def test_generation_failure_retries_same_batch_without_advancing(self):
        checkpoint = self.seed(2)
        self.store.set("session", "Previous", checkpoint)
        last_id = self.seed(2)
        self.summarizer.summarize.side_effect = RuntimeError("Model unavailable")
        with self.assertLogs("turn_finalizer", level="ERROR"):
            self.finalizer.finalize("session")
        failed_input = self.summarizer.summarize.call_args
        self.assertEqual(self.store.get("session"), ("Previous", checkpoint))
        self.summarizer.summarize.side_effect = None
        self.finalizer.finalize("session")
        self.assertEqual(self.summarizer.summarize.call_args, failed_input)
        self.assertEqual(self.store.get("session")[1], last_id)

    def test_messages_arriving_during_generation_are_left_for_next_batch(self):
        checkpoint = self.seed(2)
        def summarize(*args, **kwargs):
            self.seed(2, role="assistant")
            return "First batch"
        self.summarizer.summarize.side_effect = summarize
        self.finalizer.finalize("session")
        self.assertEqual(self.store.get("session")[1], checkpoint)
        self.summarizer.summarize.side_effect = None
        self.finalizer.finalize("session")
        self.assertEqual([row["role"] for row in self.summarizer.summarize.call_args.args[0]],
                         ["assistant", "assistant"])

    def test_legacy_summary_is_preserved_and_replayed_once_on_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "legacy.db")
            with sqlite3.connect(path) as conn:
                conn.execute("""CREATE TABLE conversation_summary (
                    session_id TEXT PRIMARY KEY, summary TEXT NOT NULL,
                    last_turn_count INTEGER DEFAULT 0, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
                conn.execute("INSERT INTO conversation_summary (session_id, summary, last_turn_count) VALUES ('session', 'Old summary', 1000)")
            db = Database(path)
            try:
                store = SummaryStore(db)
                self.assertEqual(store.get("session"), ("Old summary", 0))
                store.set("session", "Merged summary", 1200)
            finally:
                db.close()
            reopened = Database(path)
            try:
                self.assertEqual(SummaryStore(reopened).get("session"), ("Merged summary", 1200))
            finally:
                reopened.close()
