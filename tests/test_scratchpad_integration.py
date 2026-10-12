import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.config import Config
from app.core import orchestrator_factory

from app.integrations import CapabilityId, IntegrationRegistry, InvocationContext, ToolCall
from app.integrations.scratchpad import ScratchpadIntegration
from app.scratchpad.store import ScratchpadStore
from app.storage.database import Database
from tests.test_scratchpad_store import FakeNotebookCollection
from tests.test_memory_store import FakeVectorStore


class ScratchpadIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.close)
        self.collection = FakeNotebookCollection()
        self.store = ScratchpadStore(self.db, self.collection, "astra")
        self.registry = IntegrationRegistry([ScratchpadIntegration(self.store)])

    def invoke(self, action, arguments, context=None):
        return self.registry.invoke(ToolCall(CapabilityId("scratchpad", action), arguments),
                                    context or InvocationContext("session-a", "current turn"))

    def test_six_tools_roundtrip_and_revision_conflict(self):
        self.assertEqual(len(self.registry.get_native_tools()), 6)
        result = self.invoke("write", {"content": "Project Kite\nCheck launch timing"})
        self.assertEqual(result.status.value, "success")
        entry = json.loads(result.content)["entry"]
        self.assertEqual(json.loads(self.invoke("read", {"id": entry["id"]}).content)["entry"], entry)
        self.assertEqual(json.loads(self.invoke("search", {"query": "launch"}).content)["returned_count"], 1)
        self.assertEqual(json.loads(self.invoke("recent", {}).content)["total"], 1)
        edited = self.invoke("edit", {"id": entry["id"], "content": "Test completed", "expected_revision": 1})
        self.assertEqual(json.loads(edited.content)["entry"]["revision"], 2)
        conflict = self.invoke("delete", {"id": entry["id"], "expected_revision": 1})
        self.assertEqual(conflict.diagnostics["category"], "revision_conflict")
        self.assertEqual(self.invoke("delete", {"id": entry["id"], "expected_revision": 2}).status.value, "success")
        self.assertEqual(self.invoke("read", {"id": entry["id"]}).diagnostics["category"], "not_found")

    def test_schema_validation_and_event_authority_prevent_mutations(self):
        for arguments in ({"content": "valid", "owner_agent_id": "other"}, {"content": "x" * 4001}, {}, {"content": " "}):
            self.assertEqual(self.invoke("write", arguments).status.value, "error")
        event = InvocationContext("session-a", "event", event_id="event-a")
        self.assertEqual(self.registry.get_native_tools(event.allowed_capabilities), [])
        self.assertEqual(self.invoke("write", {"content": "Forbidden"}, event).status.value, "denied")
        self.assertEqual(self.store.recent()["total"], 0)
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO deleted_sessions(session_id) VALUES ('session-a')")
        self.assertEqual(self.invoke("write", {"content": "Late"}).status.value, "denied")

    def test_index_failure_is_successful_save_and_search_outage_is_unavailable(self):
        with patch.object(self.collection, "upsert", side_effect=RuntimeError("offline")), self.assertLogs("scratchpad"):
            result = self.invoke("write", {"content": "Saved despite outage"})
        self.assertEqual(result.status.value, "success")
        self.assertFalse(json.loads(result.content)["indexed"])
        with patch.object(self.collection, "count", side_effect=RuntimeError("offline")):
            self.assertEqual(self.invoke("search", {"query": "saved"}).status.value, "unavailable")
        self.assertEqual(json.loads(self.invoke("recent", {}).content)["total"], 1)

    def test_list_output_is_bounded_and_truncation_is_explicit(self):
        for index in range(20):
            self.store.write(f"{index} " + "\n" * 3900, "session-a")
        payload = json.loads(self.invoke("recent", {"limit": 20}).content)
        self.assertTrue(all(entry["truncated"] for entry in payload["entries"]))
        self.assertLessEqual(len(json.dumps(payload["entries"], ensure_ascii=False, separators=(",", ":"))), 10000)
        self.assertEqual(payload["returned_count"] + payload["omitted_count"], 20)

    def test_factory_composes_enabled_and_disabled_notebook_without_erasing_notes(self):
        self.store.write("Existing note", "session-a")
        vectors = FakeVectorStore()
        vectors.scratchpad_collection = self.collection
        vectors.close = Mock()
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.NamedTemporaryFile("w", suffix=".yaml") as file:
                file.write(f"assistant:\n  id: astra\nintegrations:\n  scratchpad:\n    enabled: {str(enabled).lower()}\n")
                file.flush()
                config = Config(file.name)
                with patch.object(orchestrator_factory, "Database", return_value=self.db), \
                        patch.object(orchestrator_factory, "VectorStore", return_value=vectors), \
                        patch.object(orchestrator_factory, "OllamaClient"), \
                        patch.object(orchestrator_factory, "AutonomyStore", return_value=SimpleNamespace(close=Mock())), \
                        patch.object(orchestrator_factory, "AutonomyRuntime"):
                    orchestrator = orchestrator_factory.build_orchestrator(config)
                tools = orchestrator.tool_executor.get_native_tools()
                notebook_tools = [tool for tool in tools if tool["function"]["name"].startswith("scratchpad__")]
                self.assertEqual(len(notebook_tools), 6 if enabled else 0)
                self.assertEqual(orchestrator.scratchpad_context_provider is not None, enabled)
                if enabled:
                    provider = orchestrator.scratchpad_context_provider
                    self.assertIs(provider.store.collection, self.collection)
                    self.assertEqual(provider.store.owner_agent_id, "astra")
                self.assertEqual(self.store.recent()["total"], 1)


if __name__ == "__main__":
    unittest.main()
