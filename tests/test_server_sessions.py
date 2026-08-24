import importlib
import base64
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
from app.perception.state import ImageAttachment, PerceptionState
from app.storage.database import Database
from app.core.conversation import (
    InputSource, SenderAttribution, SenderType, SessionKind, derive_relay_sender_id,
)


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
    def __init__(self, history_store, summary_store, gesture_catalog=None):
        self.history = history_store
        self.summary_store = summary_store
        self.gesture_catalog = gesture_catalog or {}
        self.agent_id = "agent-a"
        self.belief_repository = FakeBeliefRepository()


class FakeBeliefRepository:
    def __init__(self):
        self.deleted_sessions = []

    def delete_session(self, owner_agent_id, session_id):
        self.deleted_sessions.append((owner_agent_id, session_id))
        return 0


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


class FakeApprovalWebSocket(FakeWebSocket):
    def __init__(self, approved=True):
        super().__init__()
        self.approved = approved

    async def receive(self):
        approval_id = self.messages[-1]["approval_id"]
        return {
            "text": json.dumps({
                "type": "tool_approval_response",
                "approval_id": approval_id,
                "approved": self.approved,
            })
        }


class InvalidSttAudioErrorTests(unittest.TestCase):
    def test_pyav_invalid_data_error_is_expected_stt_silence(self):
        invalid_data_error_type = type(
            "InvalidDataError",
            (Exception,),
            {"__module__": "av.error"},
        )
        exc = invalid_data_error_type(
            "[Errno 1094995529] Invalid data found when processing input: '<none>'"
        )

        self.assertTrue(server_module._is_invalid_stt_audio_error(exc))

    def test_other_errors_are_not_treated_as_stt_silence(self):
        self.assertFalse(
            server_module._is_invalid_stt_audio_error(RuntimeError("model unavailable"))
        )


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
            gesture_catalog={"greeting": "/static/animations/Gestures/Greeting.fbx"},
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

    def test_list_sessions_excludes_repeatedly_created_empty_sessions(self):
        for index in range(3):
            direct_id = f"unused-direct-{index}"
            group_id = f"unused-group-{index}"
            for _ in range(2):
                self.history.ensure_session(direct_id, SessionKind.DIRECT)
                self.history.ensure_session(group_id, SessionKind.MANUAL_GROUP)

        sessions = server_module.asyncio.run(server_module.list_sessions())["sessions"]
        self.assertEqual(
            {session["session_id"] for session in sessions},
            {"session-a", "session-b"},
        )
        self.assertEqual(
            self.history.get_session_kind("unused-group-2"),
            SessionKind.MANUAL_GROUP,
        )
        self.assertTrue(self.history.session_exists("unused-direct-2"))

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
        self.assertEqual(
            self.fake_orchestrator.belief_repository.deleted_sessions,
            [("agent-a", "session-b")],
        )

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
        text, reasoning, instant_mode, attachments = server_module.parse_user_message(
            '{"type":"user_message","text":"hello","reasoning":true}'
        )

        self.assertEqual(text, "hello")
        self.assertIs(reasoning, True)
        self.assertIs(instant_mode, False)
        self.assertEqual(attachments, [])

    def test_parse_user_message_supports_instant_mode(self):
        text, reasoning, instant_mode, attachments = server_module.parse_user_message(
            '{"type":"user_message","text":"hello","instant_mode":true}'
        )

        self.assertEqual(text, "hello")
        self.assertIsNone(reasoning)
        self.assertIs(instant_mode, True)
        self.assertEqual(attachments, [])

    def test_parse_user_message_parses_base64_image_attachments(self):
        text, reasoning, instant_mode, attachments = server_module.parse_user_message(
            '{"type":"user_message","text":"look","attachments":[{"name":"cat.png","mime_type":"image/png","data":"aGVsbG8=","size_bytes":5}]}'
        )

        self.assertEqual(text, "look")
        self.assertIsNone(reasoning)
        self.assertIs(instant_mode, False)
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].name, "cat.png")
        self.assertEqual(attachments[0].mime_type, "image/png")
        self.assertEqual(attachments[0].base64_data, "aGVsbG8=")

    def test_append_recent_vision_attachments_adds_screen_and_webcam_frames(self):
        perception = PerceptionState()
        perception.update(
            server_module.PerceptionKey.SCREEN_SCENE,
            {
                "name": "screen.jpg",
                "mime_type": "image/jpeg",
                "base64_data": "aGVsbG8=",
            },
        )
        perception.update(
            server_module.PerceptionKey.WEBCAM_SCENE,
            {
                "name": "webcam.jpg",
                "mime_type": "image/jpeg",
                "base64_data": "d29ybGQ=",
            },
        )
        orchestrator = types.SimpleNamespace(perception=perception)
        attachments = []

        appended_count = server_module._append_recent_vision_attachments(
            orchestrator,
            attachments,
            max_age_seconds=10.0,
        )

        self.assertEqual(appended_count, 2)
        self.assertEqual([attachment.name for attachment in attachments], ["screen.jpg", "webcam.jpg"])
        self.assertEqual([attachment.base64_data for attachment in attachments], ["aGVsbG8=", "d29ybGQ="])

    def test_append_recent_vision_attachments_ignores_stale_frames(self):
        perception = PerceptionState()
        perception.update(
            server_module.PerceptionKey.SCREEN_SCENE,
            {
                "name": "screen.jpg",
                "mime_type": "image/jpeg",
                "base64_data": "aGVsbG8=",
            },
        )
        orchestrator = types.SimpleNamespace(perception=perception)
        attachments = []

        appended_count = server_module._append_recent_vision_attachments(
            orchestrator,
            attachments,
            max_age_seconds=-1.0,
        )

        self.assertEqual(appended_count, 0)
        self.assertEqual(attachments, [])

    def test_dedupe_attachments_by_hash_keeps_first_copy(self):
        first = server_module.attachment_from_payload({
            "name": "screen-a.jpg",
            "mime_type": "image/jpeg",
            "data": "aGVsbG8=",
        })
        duplicate = server_module.attachment_from_payload({
            "name": "screen-b.jpg",
            "mime_type": "image/jpeg",
            "data": "aGVsbG8=",
        })
        different = server_module.attachment_from_payload({
            "name": "screen-c.jpg",
            "mime_type": "image/jpeg",
            "data": "d29ybGQ=",
        })

        deduped = server_module._dedupe_attachments_by_hash([
            first,
            duplicate,
            different,
        ])

        self.assertEqual(
            [attachment.name for attachment in deduped],
            ["screen-a.jpg", "screen-c.jpg"],
        )

    def test_parse_user_message_repairs_prefixed_png_clipboard_payload(self):
        broken_png = b"\xbbK\xe0\x00" + b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        broken_png_b64 = base64.b64encode(broken_png).decode("ascii")

        text, reasoning, instant_mode, attachments = server_module.parse_user_message(
            '{"type":"user_message","text":"look","attachments":[{"name":"image.png","mime_type":"image/png","data":"'
            + broken_png_b64
            + '"}]}'
        )

        self.assertEqual(text, "look")
        self.assertIsNone(reasoning)
        self.assertIs(instant_mode, False)
        self.assertEqual(len(attachments), 1)
        decoded = base64.b64decode(attachments[0].base64_data)
        self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_parse_user_message_rejects_empty_payload_without_text_or_images(self):
        with self.assertRaises(ValueError):
            server_module.parse_user_message(
                '{"type":"user_message","text":"   ","attachments":[]}'
            )

    def test_parse_user_message_keeps_plain_text_backward_compatible(self):
        text, reasoning, instant_mode, attachments = server_module.parse_user_message("hello")

        self.assertEqual(text, "hello")
        self.assertIsNone(reasoning)
        self.assertIs(instant_mode, False)
        self.assertEqual(attachments, [])

    def test_relay_validation_kind_types_spoofing_and_deterministic_id(self):
        payload = {
            "type": "relay_message",
            "sender_display_name": "  Claude   Agent  ",
            "sender_type": "external_agent",
            "text": "Hello from elsewhere",
        }
        text, sender = server_module.parse_relay_message(payload, SessionKind.MANUAL_GROUP)
        self.assertEqual(text, "Hello from elsewhere")
        self.assertEqual(sender.sender_display_name, "Claude Agent")
        self.assertEqual(sender.sender_type, SenderType.EXTERNAL_AGENT)
        self.assertEqual(
            sender.sender_id,
            derive_relay_sender_id(SenderType.EXTERNAL_AGENT, "claude agent"),
        )
        human_payload = {**payload, "sender_type": "human", "sender_display_name": "Alice"}
        _, human_sender = server_module.parse_relay_message(human_payload, SessionKind.MANUAL_GROUP)
        self.assertEqual(human_sender.sender_type, SenderType.HUMAN)
        with self.assertRaises(ValueError):
            server_module.parse_relay_message(payload, SessionKind.DIRECT)
        for rejected_type in ("local_assistant", "system", "tool", "integration_runtime"):
            with self.subTest(rejected_type=rejected_type), self.assertRaises(ValueError):
                server_module.parse_relay_message(
                    {**payload, "sender_type": rejected_type}, SessionKind.MANUAL_GROUP
                )
        for spoof in ("sender_id", "role", "input_source", "target", "attachments", "tool"):
            with self.subTest(spoof=spoof), self.assertRaises(ValueError):
                server_module.parse_relay_message(
                    {**payload, spoof: "spoofed"}, SessionKind.MANUAL_GROUP
                )

    def test_user_message_rejects_server_controlled_sender_fields(self):
        for field in ("role", "sender_id", "sender_type", "input_source", "target", "tool"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                server_module.parse_user_message(json.dumps({
                    "type": "user_message", "text": "hello", field: "spoofed",
                }))

    def test_durable_session_kind_and_sender_history_api_round_trip(self):
        self.history.ensure_session("empty-group", SessionKind.MANUAL_GROUP)
        empty_payload = server_module.asyncio.run(server_module.get_session("empty-group"))
        self.assertEqual(empty_payload["kind"], "manual_group")
        self.assertEqual(empty_payload["messages"], [])

        self.assertEqual(
            self.history.ensure_session("group-api", SessionKind.MANUAL_GROUP),
            SessionKind.MANUAL_GROUP,
        )
        # Re-ensuring cannot mutate an existing session kind.
        self.assertEqual(
            self.history.ensure_session("group-api", SessionKind.DIRECT),
            SessionKind.MANUAL_GROUP,
        )
        sender = SenderAttribution(
            "relay:human:abc", "Alice", SenderType.HUMAN, InputSource.MANUAL_RELAY
        )
        self.history.add("group-api", "user", "Relayed hello", sender=sender)

        payload = server_module.asyncio.run(server_module.get_session("group-api"))
        self.assertEqual(payload["kind"], "manual_group")
        self.assertEqual(
            {key: payload["messages"][0][key] for key in (
                "sender_id", "sender_display_name", "sender_type", "input_source"
            )},
            {
                "sender_id": sender.sender_id,
                "sender_display_name": "Alice",
                "sender_type": "human",
                "input_source": "manual_relay",
            },
        )
        sessions = server_module.asyncio.run(server_module.list_sessions())["sessions"]
        self.assertEqual(next(item for item in sessions if item["session_id"] == "group-api")["kind"], "manual_group")

    def test_should_forward_state_holds_responding_until_audio(self):
        self.assertFalse(server_module._should_forward_state(server_module.AssistantState.RESPONDING))
        self.assertTrue(server_module._should_forward_state(server_module.AssistantState.THINKING))

    def test_build_session_init_payload_includes_gesture_catalog(self):
        payload = server_module._build_session_init_payload(
            server_instance_id="server-1",
            session_id="session-a",
            gesture_catalog={"greeting": "/static/animations/Gestures/Greeting.fbx"},
        )

        self.assertEqual(
            payload,
            {
                "type": "session_init",
                "server_instance_id": "server-1",
                "session_id": "session-a",
                "gesture_catalog": {"greeting": "/static/animations/Gestures/Greeting.fbx"},
                "session_kind": "direct",
                "local_human_display_name": "You",
                "local_assistant_display_name": "Astra",
            },
        )

    def test_build_attachment_drop_notice_payload_returns_none_when_no_drop(self):
        orchestrator = types.SimpleNamespace(
            llm=types.SimpleNamespace(
                last_stream_dropped_current_images=False,
                last_stream_dropped_current_images_count=0,
            )
        )

        payload = server_module._build_attachment_drop_notice_payload(
            orchestrator=orchestrator,
            original_attachment_count=1,
        )

        self.assertIsNone(payload)

    def test_build_attachment_drop_notice_payload_describes_dropped_images(self):
        orchestrator = types.SimpleNamespace(
            llm=types.SimpleNamespace(
                last_stream_dropped_current_images=True,
                last_stream_dropped_current_images_count=2,
            )
        )

        payload = server_module._build_attachment_drop_notice_payload(
            orchestrator=orchestrator,
            original_attachment_count=2,
        )

        self.assertEqual(
            payload,
            {
                "type": "user_notice",
                "scope": "last_user_message",
                "tone": "warning",
                "message": "Attached images were not sent to the model for this message.",
            },
        )

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

    def test_request_tool_approval_sends_prompt_and_returns_response(self):
        ws = FakeApprovalWebSocket(approved=True)

        approved = server_module.asyncio.run(
            server_module._request_tool_approval(
                ws,
                {
                    "tool": "shell__execute",
                    "title": "Approve command?",
                    "reason": "Command requires approval.",
                    "detail_label": "Command",
                    "detail": "printf hi",
                },
                connection_id="conn-1",
                timeout_seconds=1.0,
            )
        )

        self.assertTrue(approved)
        self.assertEqual(ws.messages[0]["type"], "tool_approval_request")
        self.assertEqual(ws.messages[0]["tool"], "shell__execute")
        self.assertEqual(ws.messages[0]["title"], "Approve command?")
        self.assertEqual(ws.messages[0]["detail_label"], "Command")
        self.assertEqual(ws.messages[0]["detail"], "printf hi")
        self.assertEqual(ws.messages[0]["reason"], "Command requires approval.")

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

    def test_prepare_tts_text_drops_complete_thinking_blocks(self):
        text = "Before <think>\nsecret plan\n</think>\n\nAfter."

        prepared = server_module._prepare_tts_text(text)

        self.assertEqual(prepared, "Before After.")

    def test_thinking_block_filter_strips_streamed_reasoning_across_chunk_boundaries(self):
        filter_ = server_module.ThinkingBlockFilter()
        chunks = [
            "Hello ",
            "<thi",
            "nk>\nReasoning sentence. Another one.",
            "\n</th",
            "ink>\n\nworld.",
        ]

        visible = "".join(filter_.push(chunk) for chunk in chunks) + filter_.flush()

        self.assertEqual(visible, "Hello \n\nworld.")


if __name__ == "__main__":
    unittest.main()
