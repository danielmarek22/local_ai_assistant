import importlib
import sys
import types
import unittest
from unittest.mock import patch

from app.memory.chat_history import ChatHistoryStore
from app.memory.summary_store import SummaryStore
from app.storage.database import Database


def _load_server_module():
    if "app.server" in sys.modules:
        return sys.modules["app.server"]

    fake_piper_module = types.ModuleType("piper")
    fake_piper_config_module = types.ModuleType("piper.config")

    class FakePiperVoice:
        @staticmethod
        def load(_model_path, use_cuda=True):
            return FakePiperVoice()

        def synthesize_wav(self, text, wav_file, syn_config=None):
            return None

    class FakeSynthesisConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_piper_module.PiperVoice = FakePiperVoice
    fake_piper_config_module.SynthesisConfig = FakeSynthesisConfig

    with patch.dict(
        sys.modules,
        {
            "piper": fake_piper_module,
            "piper.config": fake_piper_config_module,
        },
    ):
        return importlib.import_module("app.server")


server_module = _load_server_module()


class FakeOrchestrator:
    def __init__(self, history_store, summary_store, session_id="active-session"):
        self.history = history_store
        self.summary_store = summary_store
        self.session_id = session_id
        self.session_switches = []

    def set_session(self, session_id: str):
        self.session_id = session_id
        self.session_switches.append(session_id)

class FakeCollection:
    def __init__(self): self.docs = []
    def add(self, *args, **kwargs): pass
    def query(self, *args, **kwargs): return {"documents": [[]]}

class FakeVectorStore:
    def __init__(self):
        self.semantic_collection = FakeCollection()
        self.episodic_collection = FakeCollection()


class ServerSessionTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(path=":memory:")
        self.vector_store = FakeVectorStore() # NEW
        self.history = ChatHistoryStore(self.db, self.vector_store) # UPDATED
        self.summary_store = SummaryStore(self.db)

        self.history.add("session-a", "user", "First chat")
        self.history.add("session-a", "assistant", "Reply A")
        self.history.add("session-b", "user", "Second chat")
        self.history.add("session-b", "assistant", "Reply B")
        self.summary_store.set("session-b", "Summary B")

        self.fake_orchestrator = FakeOrchestrator(
            history_store=self.history,
            summary_store=self.summary_store,
            session_id="session-b",
        )
        server_module.app.state.orchestrator = self.fake_orchestrator
        server_module.app.state.server_instance_id = "server-1"

    def test_list_sessions_returns_saved_sessions(self):
        response = server_module.asyncio.run(server_module.list_sessions())

        sessions = response["sessions"]

        self.assertEqual({session["session_id"] for session in sessions}, {"session-a", "session-b"})
        session_b = next(session for session in sessions if session["session_id"] == "session-b")
        self.assertEqual(session_b["message_count"], 2)
        self.assertEqual(session_b["preview"], "Second chat")

    def test_get_session_returns_messages_and_summary(self):
        payload = server_module.asyncio.run(server_module.get_session("session-b"))

        self.assertEqual(payload["session_id"], "session-b")
        self.assertEqual(payload["summary"], "Summary B")
        self.assertEqual(
            [(message["role"], message["content"]) for message in payload["messages"]],
            [
                ("user", "Second chat"),
                ("assistant", "Reply B"),
            ],
        )

    def test_delete_session_removes_rows_and_resets_active_session(self):
        response = server_module.asyncio.run(server_module.delete_session("session-b"))

        self.assertEqual(response["deleted"], True)
        self.assertEqual(self.history.get_all("session-b"), [])
        self.assertIsNone(self.summary_store.get("session-b"))
        self.assertEqual(len(self.fake_orchestrator.session_switches), 1)
        self.assertNotEqual(self.fake_orchestrator.session_switches[0], "session-b")

    def test_resolve_session_id_uses_existing_session_when_server_matches(self):
        session_id = server_module.resolve_session_id(
            session_mode="resume",
            requested_session_id="session-a",
            known_server_instance_id="server-1",
            server_instance_id="server-1",
        )

        self.assertEqual(session_id, "session-a")

    def test_resolve_session_id_reopens_requested_session_in_open_mode(self):
        session_id = server_module.resolve_session_id(
            session_mode="open",
            requested_session_id="session-a",
            known_server_instance_id="stale-server",
            server_instance_id="server-1",
        )

        self.assertEqual(session_id, "session-a")

    def test_resolve_session_id_ignores_stale_server_instance_for_resume(self):
        session_id = server_module.resolve_session_id(
            session_mode="resume",
            requested_session_id="session-a",
            known_server_instance_id="stale-server",
            server_instance_id="server-1",
        )

        self.assertNotEqual(session_id, "session-a")

    def test_parse_user_message_supports_structured_reasoning_override(self):
        text, reasoning = server_module.parse_user_message(
            '{"type":"user_message","text":"hello","reasoning":true}'
        )

        self.assertEqual(text, "hello")
        self.assertIs(reasoning, True)

    def test_parse_user_message_keeps_plain_text_backward_compatible(self):
        text, reasoning = server_module.parse_user_message("hello")

        self.assertEqual(text, "hello")
        self.assertIsNone(reasoning)


if __name__ == "__main__":
    unittest.main()
