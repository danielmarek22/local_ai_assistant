import importlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from app.memory.chat_history import ChatHistoryStore
from app.memory.summary_store import SummaryStore
from app.perception.state import ImageAttachment
from app.storage.database import Database


def _load_server_module():
    if "app.server" in sys.modules:
        return sys.modules["app.server"]

    fake_qwen_module = types.ModuleType("qwen_tts")
    fake_soundfile_module = types.ModuleType("soundfile")
    fake_torch_module = types.ModuleType("torch")

    class FakeQwen3TTSModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeQwen3TTSModel()

        def create_voice_clone_prompt(self, **kwargs):
            return {"prompt": "ok"}

        def generate_voice_clone(self, **kwargs):
            return [[0.0]], 24000

        def generate_custom_voice(self, **kwargs):
            return [[0.0]], 24000

    def fake_sf_read(_path):
        return [0.0], 24000

    def fake_sf_write(_path, _audio, _sr):
        return None

    fake_qwen_module.Qwen3TTSModel = FakeQwen3TTSModel
    fake_soundfile_module.read = fake_sf_read
    fake_soundfile_module.write = fake_sf_write
    fake_torch_module.bfloat16 = object()

    with patch.dict(
        sys.modules,
        {
            "qwen_tts": fake_qwen_module,
            "soundfile": fake_soundfile_module,
            "torch": fake_torch_module,
        },
    ):
        return importlib.import_module("app.server")


server_module = _load_server_module()


class FakeOrchestrator:
    def __init__(self, history_store, summary_store):
        self.history = history_store
        self.summary_store = summary_store


class FakeMemoryReflector:
    def __init__(self):
        self.calls = []
        self.next_result = {
            "success": True,
            "days_old": 0,
            "stale_count": 3,
            "deleted_count": 1,
            "kept_count": 2,
            "created_count": 1,
            "delete_ids": ["mem-1"],
            "keep_ids": ["mem-2", "mem-3"],
            "new_memories": [
                {
                    "content": "Consolidated preference",
                    "category": "preference",
                    "importance": 3,
                }
            ],
            "error": None,
        }

    def reflect_and_prune(self, days_old):
        self.calls.append(days_old)
        return self.next_result


class FakeCollection:
    def __init__(self):
        self.docs = []
        self.deleted_wheres = []

    def add(self, *args, **kwargs):
        pass

    def query(self, *args, **kwargs):
        return {"documents": [[]]}

    def delete(self, where=None):
        self.deleted_wheres.append(where)

class FakeVectorStore:
    def __init__(self):
        self.semantic_collection = FakeCollection()
        self.episodic_collection = FakeCollection()


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, payload: str):
        self.messages.append(json.loads(payload))


class ServerSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.db = Database(path=":memory:")
        self.vector_store = FakeVectorStore()
        self.history = ChatHistoryStore(self.db, self.vector_store, uploads_root=self.temp_dir.name)
        self.summary_store = SummaryStore(self.db)

        self.history.add("session-a", "user", "First chat")
        self.history.add("session-a", "assistant", "Reply A")
        self.history.add("session-b", "user", "Second chat")
        self.history.add("session-b", "assistant", "Reply B")

        self.summary_store.set("session-b", "Summary B", 2)

        self.fake_orchestrator = FakeOrchestrator(
            history_store=self.history,
            summary_store=self.summary_store,
        )
        self.fake_reflector = FakeMemoryReflector()
        server_module.app.state.orchestrator = self.fake_orchestrator
        server_module.app.state.server_instance_id = "server-1"
        server_module.app.state.memory_reflector = self.fake_reflector

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

    def test_get_session_includes_stored_image_attachments(self):
        self.history.add(
            "session-images",
            "user",
            "Screenshot here",
            attachments=[
                ImageAttachment(
                    name="screen.png",
                    mime_type="image/png",
                    base64_data="aGVsbG8=",
                    size_bytes=5,
                )
            ],
        )

        payload = server_module.asyncio.run(server_module.get_session("session-images"))

        self.assertEqual(len(payload["messages"]), 1)
        attachments = payload["messages"][0]["attachments"]
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]["name"], "screen.png")
        self.assertEqual(attachments[0]["mime_type"], "image/png")
        self.assertTrue(attachments[0]["url"].startswith("/static/"))

    def test_delete_session_removes_rows(self):
        message_id = self.history.add(
            "session-b",
            "user",
            "Attached image",
            attachments=[
                ImageAttachment(
                    name="screen.png",
                    mime_type="image/png",
                    base64_data="aGVsbG8=",
                    size_bytes=5,
                )
            ],
        )
        attachment_dir = Path(self.temp_dir.name) / "session-b" / str(message_id)
        self.assertTrue(attachment_dir.exists())

        response = server_module.asyncio.run(server_module.delete_session("session-b"))

        self.assertEqual(response["deleted"], True)
        self.assertEqual(self.history.get_all("session-b"), [])
        self.assertIsNone(self.summary_store.get("session-b"))
        self.assertIn(
            {"session_id": "session-b"},
            self.vector_store.episodic_collection.deleted_wheres,
        )
        self.assertFalse(attachment_dir.exists())

    def test_run_memory_reflection_returns_reflector_summary(self):
        payload = server_module.asyncio.run(
            server_module.run_memory_reflection(
                server_module.ReflectRequest(days_old=0)
            )
        )

        self.assertEqual(self.fake_reflector.calls, [0])
        self.assertEqual(payload["days_old"], 0)
        self.assertEqual(payload["stale_count"], 3)
        self.assertEqual(payload["deleted_count"], 1)
        self.assertEqual(payload["created_count"], 1)

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
        text, reasoning, attachments = server_module.parse_user_message(
            '{"type":"user_message","text":"hello","reasoning":true}'
        )

        self.assertEqual(text, "hello")
        self.assertIs(reasoning, True)
        self.assertEqual(attachments, [])

    def test_parse_user_message_parses_base64_image_attachments(self):
        text, reasoning, attachments = server_module.parse_user_message(
            '{"type":"user_message","text":"look","attachments":[{"name":"cat.png","mime_type":"image/png","data":"aGVsbG8=","size_bytes":5}]}'
        )

        self.assertEqual(text, "look")
        self.assertIsNone(reasoning)
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].name, "cat.png")
        self.assertEqual(attachments[0].mime_type, "image/png")
        self.assertEqual(attachments[0].base64_data, "aGVsbG8=")

    def test_parse_user_message_rejects_empty_payload_without_text_or_images(self):
        with self.assertRaises(ValueError):
            server_module.parse_user_message(
                '{"type":"user_message","text":"   ","attachments":[]}'
            )

    def test_parse_user_message_keeps_plain_text_backward_compatible(self):
        text, reasoning, attachments = server_module.parse_user_message("hello")

        self.assertEqual(text, "hello")
        self.assertIsNone(reasoning)
        self.assertEqual(attachments, [])

    def test_should_forward_state_holds_responding_until_audio(self):
        self.assertFalse(server_module._should_forward_state(server_module.AssistantState.RESPONDING))
        self.assertTrue(server_module._should_forward_state(server_module.AssistantState.THINKING))

    def test_flush_pending_chunks_sends_buffered_text_in_order(self):
        ws = FakeWebSocket()
        pending_chunks = ["Hello", " world"]

        server_module.asyncio.run(server_module._flush_pending_chunks(ws, pending_chunks))

        self.assertEqual(
            ws.messages,
            [
                {"type": "assistant_chunk", "content": "Hello"},
                {"type": "assistant_chunk", "content": " world"},
            ],
        )
        self.assertEqual(pending_chunks, [])

    def test_prepare_tts_text_removes_markdown_blocks_and_markers(self):
        text = (
            "# Title\n"
            "- First item\n"
            "1. Second item\n"
            "> Quoted line\n"
            "---\n"
            "After."
        )

        prepared = server_module._prepare_tts_text(text)

        self.assertEqual(
            prepared,
            "Title First item Second item Quoted line After.",
        )

    def test_prepare_tts_text_keeps_labels_and_drops_markdown_urls(self):
        text = (
            "Read [docs](https://example.com/docs) and "
            "![logo](https://example.com/logo.png). "
            "Reference [guide][guide-ref]. "
            "<https://example.com/raw>\n"
            "[guide-ref]: https://example.com/guide"
        )

        prepared = server_module._prepare_tts_text(text)

        self.assertEqual(prepared, "Read docs and logo. Reference guide.")

    def test_prepare_tts_text_drops_fenced_code_and_unwraps_inline_styles(self):
        text = (
            "Use **bold**, _italic_, `inline code`, and ~~strikethrough~~.\n"
            "```python\n"
            "print('hidden')\n"
            "```\n"
            "Escaped \\*literal\\* marker."
        )

        prepared = server_module._prepare_tts_text(text)

        self.assertEqual(
            prepared,
            "Use bold, italic, inline code, and strikethrough. Escaped *literal* marker.",
        )


if __name__ == "__main__":
    unittest.main()
