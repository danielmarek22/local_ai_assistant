import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.session_ids import SessionDeletedError
from app.scratchpad.store import ScratchpadStore, ScratchpadConflict, ScratchpadNotFound
from app.storage.database import Database


class FakeNotebookCollection:
    def __init__(self):
        self.records = {}
        self.queries = []

    def upsert(self, ids, documents, metadatas):
        for identity, document, metadata in zip(ids, documents, metadatas):
            self.records[identity] = (document, metadata, 0.2)

    def count(self):
        return len(self.records)

    def get(self, where=None, include=None):
        return {"ids": [identity for identity, (_, metadata, _) in self.records.items()
                        if where is None or all(metadata.get(k) == v for k, v in where.items())]}

    def query(self, query_texts, n_results, where=None, include=None):
        self.queries.append(query_texts[0])
        ids = self.get(where)["ids"][:n_results]
        return {"ids": [ids], "metadatas": [[self.records[i][1] for i in ids]],
                "distances": [[self.records[i][2] for i in ids]]}

    def delete(self, ids):
        for identity in ids:
            self.records.pop(identity, None)


class ScratchpadStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.close)
        self.collection = FakeNotebookCollection()
        self.store = ScratchpadStore(self.db, self.collection, "astra")

    def test_crud_revision_pagination_and_owner_isolation(self):
        entry = self.store.write("# Project\nKeep the internal formatting.\n", "session-a")["entry"]
        self.assertEqual(self.store.read(entry["id"]), entry)
        other = ScratchpadStore(self.db, self.collection, "other-agent")
        other.write("Other owner's note", "session-b")
        with self.assertRaises(ScratchpadNotFound):
            other.read(entry["id"])
        with self.assertRaises(ScratchpadNotFound):
            other.edit(entry["id"], "no", 1, "session-b")
        self.assertEqual(len(self.store.search("project")), 1)
        updated = self.store.edit(entry["id"], "Corrected", 1, "session-b")["entry"]
        self.assertEqual(updated["revision"], 2)
        self.assertEqual(updated["created_at"], entry["created_at"])
        for mutate in (lambda: self.store.edit(entry["id"], "Stale", 1, "session-b"),
                       lambda: self.store.delete(entry["id"], 1, "session-b")):
            with self.assertRaises(ScratchpadConflict):
                mutate()
        self.store.write("Second", "session-b")
        page = self.store.recent(1)
        self.assertEqual(page["total"], 2)
        self.assertNotEqual(page["entries"][0]["id"], self.store.recent(1, 1)["entries"][0]["id"])
        self.store.delete(entry["id"], 2, "session-b")
        with self.assertRaises(ScratchpadNotFound):
            self.store.read(entry["id"])
        self.assertEqual(len(other.search("other")), 1)

    def test_reopen_and_session_deletion_keep_independent_notes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "notebook.db")
            db = Database(path)
            entry = ScratchpadStore(db, self.collection, "astra").write("Project Kite", "session-a")["entry"]
            db.close()
            db = Database(path)
            try:
                store = ScratchpadStore(db, self.collection, "astra")
                with db.transaction() as conn:
                    conn.execute("INSERT INTO deleted_sessions(session_id) VALUES ('session-a')")
                self.assertEqual(store.read(entry["id"])["content"], "Project Kite")
                for mutate in (lambda: store.write("Late", "session-a"),
                               lambda: store.edit(entry["id"], "Late", 1, "session-a"),
                               lambda: store.delete(entry["id"], 1, "session-a")):
                    with self.assertRaises(SessionDeletedError):
                        mutate()
                self.assertEqual(store.edit(entry["id"], "Changed", 1, "session-b")["entry"]["revision"], 2)
            finally:
                db.close()

    def test_index_failure_reports_saved_and_filters_old_revisions_until_repair(self):
        entry = self.store.write("Old topic", "session-a")["entry"]
        with patch.object(self.collection, "upsert", side_effect=RuntimeError("offline")), self.assertLogs("scratchpad"):
            result = self.store.edit(entry["id"], "Entirely new topic", 1, "session-a")
            added = self.store.write("Unindexed", "session-a")
        self.assertTrue(result["saved"])
        self.assertFalse(result["indexed"])
        self.assertIn("Do not repeat", result["warning"])
        self.assertEqual(self.store.read(entry["id"])["content"], "Entirely new topic")
        self.assertEqual(self.store.recent()["total"], 2)
        self.assertEqual(self.store.search("old topic"), [])
        other = ScratchpadStore(self.db, self.collection, "other").write("Keep", "session-b")["entry"]
        self.collection.upsert(["orphan"], ["No row"], [{"owner_agent_id": "astra", "revision": 1}])
        report = self.store.reconcile_index()
        self.assertEqual(report["deleted_orphan_count"], 1)
        self.assertIn(other["id"], self.collection.records)
        self.assertEqual({row["id"] for row in self.store.search("topic")}, {entry["id"], added["entry"]["id"]})
        with patch.object(self.collection, "delete", side_effect=RuntimeError("offline")), self.assertLogs("scratchpad"):
            deleted = self.store.delete(entry["id"], 2, "session-b")
        self.assertTrue(deleted["deleted"])
        self.assertFalse(deleted["indexed"])
        self.assertNotIn(entry["id"], {row["id"] for row in self.store.search("topic")})

    def test_invalid_inputs_and_candidate_authority(self):
        for content in ("   ", "x" * 4001, None):
            with self.assertRaises(ValueError):
                self.store.write(content, "session-a")
        for limit in (0, 21, True):
            with self.assertRaises(ValueError):
                self.store.recent(limit)
        entry = self.store.write("Canonical text", "session-a")["entry"]
        for distance, metadata in (
            (0.9, {"owner_agent_id": "astra", "revision": 1}),
            (float("nan"), {"owner_agent_id": "astra", "revision": 1}),
            (0.1, {"owner_agent_id": "foreign", "revision": 1}),
            (0.1, {"owner_agent_id": "astra", "revision": True}),
            (0.1, {"owner_agent_id": "astra", "revision": 9}),
        ):
            with patch.object(self.collection, "query", return_value={
                "ids": [[entry["id"]]], "distances": [[distance]], "metadatas": [[metadata]],
                "documents": [["Injected vector text"]],
            }):
                self.assertEqual(self.store.search("query"), [])
        self.collection.records[entry["id"]] = ("Untrusted vector text", {"owner_agent_id": "astra", "revision": 1}, 0.1)
        self.assertEqual(self.store.search("query")[0]["content"], "Canonical text")

    def test_mutations_serialize_index_updates_without_blocking_database_reads(self):
        entry = self.store.write("Original", "session-a")["entry"]
        started, release = threading.Event(), threading.Event()
        outcomes = []
        original = self.collection.upsert

        def slow_upsert(*args, **kwargs):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test timed out")
            original(*args, **kwargs)

        def edit(text):
            try:
                outcomes.append(self.store.edit(entry["id"], text, 1, "session-a"))
            except ScratchpadConflict:
                outcomes.append("conflict")

        with patch.object(self.collection, "upsert", side_effect=slow_upsert):
            first = threading.Thread(target=edit, args=("New",))
            second = threading.Thread(target=edit, args=("Stale",))
            first.start()
            try:
                self.assertTrue(started.wait(3))
                self.assertEqual(self.store.read(entry["id"])["content"], "New")
                second.start()
            finally:
                release.set()
                first.join(3)
                if second.ident is not None:
                    second.join(3)
        self.assertIn("conflict", outcomes)
        self.assertEqual(self.collection.records[entry["id"]][0], "New")


if __name__ == "__main__":
    unittest.main()
