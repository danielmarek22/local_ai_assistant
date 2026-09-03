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


RED_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
BLUE_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
THREE_BY_TWO_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAEUlEQVR4nGNk+M8AAUwMMAAAEioBAy0HqkIAAAAASUVORK5CYII="


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


class ServerLifecycleTests(unittest.TestCase):
    def _settings(self):
        return types.SimpleNamespace(
            logging={"dir": "logs"},
            raw={},
            tts={"engine": "fake"},
            stt={"enabled": True},
            voice_input={"path": "stt"},
            autonomy={"approval_timeout_s": 10},
        )

    def _orchestrator(self):
        history = types.SimpleNamespace(list_sessions=lambda: [])

        class FakeRuntimeOrchestrator:
            def __init__(self):
                self.llm = object()
                self.memory_retriever = types.SimpleNamespace(memory=object())
                self.history = history
                self.autonomy_runtime = None
                self.closed = False

            def close(self):
                self.closed = True

        return FakeRuntimeOrchestrator()

    def test_factory_owns_injected_runtime_for_its_lifespan(self):
        settings = self._settings()
        orchestrator = self._orchestrator()
        fake_tts = types.SimpleNamespace(synthesize=lambda *_args: None)
        fake_stt = object()
        application = server_module.create_app(
            settings,
            orchestrator_builder=lambda received: (
                orchestrator if received is settings else self.fail("wrong settings")
            ),
            tts_builder=lambda _config: fake_tts,
            stt_builder=lambda _config: fake_stt,
        )

        async def exercise_lifespan():
            async with application.router.lifespan_context(application):
                self.assertIs(application.state.settings, settings)
                self.assertIs(application.state.orchestrator, orchestrator)
                self.assertIs(application.state.tts, fake_tts)
                self.assertIs(application.state.stt, fake_stt)
                request = server_module.Request({"type": "http", "app": application})
                self.assertEqual(
                    await server_module.list_sessions(request),
                    {"sessions": []},
                )

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            server_module,
            "AUDIO_DIR",
            Path(temp_dir) / "audio",
        ), patch.object(server_module, "setup_logging_from_config"):
            server_module.asyncio.run(exercise_lifespan())

        self.assertTrue(orchestrator.closed)

    def test_factory_rolls_back_resources_when_startup_fails(self):
        settings = self._settings()
        orchestrator = self._orchestrator()
        fake_tts = types.SimpleNamespace(synthesize=lambda *_args: None)

        def fail_stt(_config):
            raise RuntimeError("stt failed")

        application = server_module.create_app(
            settings,
            orchestrator_builder=lambda _settings: orchestrator,
            tts_builder=lambda _config: fake_tts,
            stt_builder=fail_stt,
        )

        async def exercise_lifespan():
            async with application.router.lifespan_context(application):
                self.fail("startup failure should prevent lifespan entry")

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            server_module,
            "AUDIO_DIR",
            Path(temp_dir) / "audio",
        ), patch.object(server_module, "setup_logging_from_config"):
            with self.assertRaisesRegex(RuntimeError, "stt failed"):
                server_module.asyncio.run(exercise_lifespan())

        self.assertTrue(orchestrator.closed)
        self.assertTrue(application.state.tts_worker_task.done())

    def test_startup_removes_only_generated_audio_files(self):
        settings = self._settings()
        orchestrator = self._orchestrator()
        application = server_module.create_app(
            settings,
            orchestrator_builder=lambda _settings: orchestrator,
            tts_builder=lambda _config: types.SimpleNamespace(synthesize=lambda *_args: None),
            stt_builder=lambda _config: object(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_dir = Path(temp_dir) / "audio"
            audio_dir.mkdir()
            generated = audio_dir / f"{'a' * 32}.wav"
            unrelated = audio_dir / "reference.wav"
            nested = audio_dir / f"{'b' * 32}.wav"
            generated.write_bytes(b"generated")
            unrelated.write_bytes(b"keep")
            nested.mkdir()

            async def exercise_lifespan():
                async with application.router.lifespan_context(application):
                    self.assertFalse(generated.exists())
                    self.assertTrue(unrelated.exists())
                    self.assertTrue(nested.is_dir())

            with patch.object(server_module, "AUDIO_DIR", audio_dir), patch.object(
                server_module, "setup_logging_from_config"
            ):
                server_module.asyncio.run(exercise_lifespan())

        self.assertTrue(orchestrator.closed)

    def test_audio_cleanup_continues_after_individual_delete_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_dir = Path(temp_dir)
            first = audio_dir / f"{'a' * 32}.wav"
            second = audio_dir / f"{'b' * 32}.wav"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            original_unlink = Path.unlink

            def selective_unlink(path):
                if path == first:
                    raise OSError("busy")
                return original_unlink(path)

            with self.assertLogs("server", level="WARNING"), patch.object(
                Path, "unlink", selective_unlink
            ):
                report = server_module._cleanup_generated_audio(audio_dir)

            self.assertEqual(report, {"deleted_count": 1, "failed_count": 1})
            self.assertTrue(first.exists())
            self.assertFalse(second.exists())


class FakeCollection:
    def __init__(self):
        self.docs = []
        self.deleted_wheres = []

    def add(self, *args, **kwargs):
        pass

    def upsert(self, *args, **kwargs):
        pass

    def get(self, include=None):
        return {"ids": []}

    def query(self, *args, **kwargs):
        return {"documents": [[]]}

    def delete(self, ids=None, where=None):
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


class FailingSendWebSocket(FakeWebSocket):
    async def send_text(self, payload: str):
        raise RuntimeError("send failed")


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


class FakeInterleavedApprovalWebSocket(FakeWebSocket):
    def __init__(self, interleaved_messages):
        super().__init__()
        self.interleaved_messages = list(interleaved_messages)

    async def receive(self):
        if self.interleaved_messages:
            return self.interleaved_messages.pop(0)
        approval_id = self.messages[-1]["approval_id"]
        return {
            "text": json.dumps({
                "type": "tool_approval_response",
                "approval_id": approval_id,
                "approved": True,
            })
        }


class FakeFrameWebSocket:
    def __init__(self, message):
        self.message = message
        self.close_payload = None

    async def receive(self):
        return self.message

    async def close(self, *, code, reason):
        self.close_payload = {"code": code, "reason": reason}


class FakeHandshakeWebSocket:
    def __init__(self, query_params):
        self.query_params = query_params
        self.close_payload = None

    async def close(self, *, code, reason):
        self.close_payload = {"code": code, "reason": reason}


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

    def test_stream_failure_always_clears_active_turn_metadata(self):
        hub = server_module.SessionConnectionHub()
        ws = FailingSendWebSocket()
        ws.app = types.SimpleNamespace(
            state=types.SimpleNamespace(connection_hub=hub)
        )
        hub.register("session-a", "connection-a", ws)

        async def async_events(_iterator):
            yield server_module.AssistantThinkingEvent(text="thinking")

        with patch.object(server_module, "run_generator", async_events):
            with self.assertRaisesRegex(RuntimeError, "send failed"):
                server_module.asyncio.run(server_module._stream_orchestrator_events(
                    ws,
                    self.fake_orchestrator,
                    iter(()),
                    "connection-a",
                    0,
                    {"state": "idle"},
                ))

        connection = hub._connections["connection-a"]
        self.assertIsNone(connection.turn_id)
        self.assertIsNone(connection.turn_origin)

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

    def test_get_session_exposes_retry_state_on_original_user_message(self):
        message_id = self.history.add("retry-session", "user", "Please try")
        self.history.mark_turn_failed(
            "retry-session", message_id, "Astra couldn't finish this response."
        )

        payload = server_module.asyncio.run(server_module.get_session("retry-session"))

        self.assertEqual(payload["messages"][0]["id"], message_id)
        self.assertEqual(payload["messages"][0]["retryable_failure"], {
            "message": "Astra couldn't finish this response.",
            "attempts": 1,
        })

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
        self.assertEqual(response["cleanup_complete"], True)
        self.assertEqual(response["cleanup_errors"], [])
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

    def test_resolve_session_id_restores_saved_session_after_server_restart(self):
        session_id = server_module.resolve_session_id(
            session_mode="resume",
            requested_session_id="session-a",
            known_server_instance_id="stale-server",
            server_instance_id="server-1",
            requested_session_exists=True,
        )

        self.assertEqual(session_id, "session-a")

    def test_resolve_session_id_rejects_unsafe_requested_ids(self):
        unsafe_ids = (
            "../outside",
            "/tmp/outside",
            r"..\outside",
            ".",
            "session with spaces",
            "x" * 129,
        )

        for requested_session_id in unsafe_ids:
            with self.subTest(requested_session_id=requested_session_id):
                with self.assertRaisesRegex(ValueError, "Invalid session ID"):
                    server_module.resolve_session_id(
                        session_mode="open",
                        requested_session_id=requested_session_id,
                        known_server_instance_id="server-1",
                        server_instance_id="server-1",
                    )

    def test_websocket_rejects_unsafe_session_id_before_accepting(self):
        ws = FakeHandshakeWebSocket({
            "session_mode": "open",
            "session_id": "../outside",
        })

        server_module.asyncio.run(server_module.websocket_endpoint(ws))

        self.assertEqual(ws.close_payload["code"], 1008)
        self.assertIn("Invalid session ID", ws.close_payload["reason"])

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

    def test_parse_retry_message_accepts_only_a_stored_message_reference(self):
        message_id, reasoning, instant_mode = server_module.parse_retry_message({
            "type": "retry_message",
            "message_id": 42,
            "reasoning": False,
            "instant_mode": True,
        })

        self.assertEqual(message_id, 42)
        self.assertIs(reasoning, False)
        self.assertTrue(instant_mode)
        with self.assertRaises(ValueError):
            server_module.parse_retry_message({
                "type": "retry_message", "message_id": "42"
            })

    def test_parse_user_message_parses_base64_image_attachments(self):
        text, reasoning, instant_mode, attachments = server_module.parse_user_message(
            json.dumps({
                "type": "user_message",
                "text": "look",
                "attachments": [{
                    "name": "cat.png",
                    "mime_type": "image/png",
                    "data": RED_PNG_BASE64,
                    "size_bytes": 1,
                }],
            })
        )

        self.assertEqual(text, "look")
        self.assertIsNone(reasoning)
        self.assertIs(instant_mode, False)
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].name, "cat.png")
        self.assertEqual(attachments[0].mime_type, "image/png")
        self.assertEqual(attachments[0].base64_data, RED_PNG_BASE64)
        self.assertEqual(attachments[0].size_bytes, 69)

    def test_parse_user_message_rejects_attachment_count_before_decoding(self):
        payload = {
            "type": "user_message",
            "text": "too many",
            "attachments": [{"invalid": True} for _ in range(3)],
        }
        with self.assertRaisesRegex(ValueError, "at most 2 attachments"):
            server_module.parse_user_message(
                json.dumps(payload),
                max_attachment_count=2,
            )

    def test_parse_user_message_rejects_aggregate_decoded_size(self):
        payload = {
            "type": "user_message",
            "text": "too large together",
            "attachments": [
                {"name": "a.png", "mime_type": "image/png", "data": RED_PNG_BASE64},
                {"name": "b.png", "mime_type": "image/png", "data": BLUE_PNG_BASE64},
            ],
        }
        with self.assertRaisesRegex(ValueError, "aggregate limit"):
            server_module.parse_user_message(
                json.dumps(payload),
                max_attachment_bytes=100,
                max_total_attachment_bytes=137,
            )

    def test_attachment_rejects_oversized_base64_before_decoding(self):
        with patch("app.perception.attachments.base64.b64decode") as decode:
            with self.assertRaisesRegex(ValueError, "3-byte limit"):
                server_module.attachment_from_payload(
                    {
                        "name": "large.png",
                        "mime_type": "image/png",
                        "data": "A" * 8,
                    },
                    max_bytes=3,
                )
        decode.assert_not_called()

    def test_attachment_checks_decoded_size_at_base64_boundary(self):
        encoded = base64.b64encode(b"12345").decode("ascii")
        with self.assertRaisesRegex(ValueError, "4-byte limit"):
            server_module.attachment_from_payload(
                {
                    "name": "large.png",
                    "mime_type": "image/png",
                    "data": encoded,
                },
                max_bytes=4,
            )

    def test_attachment_rejects_invalid_client_reported_size(self):
        for value in (True, -1, "5"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "non-negative integer",
            ):
                server_module.attachment_from_payload({
                    "name": "image.png",
                    "mime_type": "image/png",
                    "data": "aGVsbG8=",
                    "size_bytes": value,
                })

    def test_attachment_rejects_mime_and_content_mismatch(self):
        with self.assertRaisesRegex(ValueError, "PNG, not JPEG"):
            server_module.attachment_from_payload({
                "name": "pretend.jpg",
                "mime_type": "image/jpeg",
                "data": RED_PNG_BASE64,
            })

    def test_attachment_rejects_corrupt_image_content(self):
        corrupt = base64.b64encode(b"\x89PNG\r\n\x1a\nnot-an-image").decode("ascii")
        with self.assertRaisesRegex(ValueError, "invalid or corrupted"):
            server_module.attachment_from_payload({
                "name": "broken.png",
                "mime_type": "image/png",
                "data": corrupt,
            })

    def test_attachment_enforces_dimension_and_pixel_limits(self):
        payload = {
            "name": "wide.png",
            "mime_type": "image/png",
            "data": THREE_BY_TWO_PNG_BASE64,
        }
        with self.assertRaisesRegex(ValueError, "side limit"):
            server_module.attachment_from_payload(payload, max_dimension=2)
        with self.assertRaisesRegex(ValueError, "5-pixel limit"):
            server_module.attachment_from_payload(
                payload,
                max_dimension=10,
                max_pixels=5,
            )

    def test_append_recent_vision_attachments_adds_screen_and_webcam_frames(self):
        perception = PerceptionState()
        perception.update(
            server_module.PerceptionKey.SCREEN_SCENE,
            {
                "name": "screen.png",
                "mime_type": "image/png",
                "base64_data": RED_PNG_BASE64,
            },
        )
        perception.update(
            server_module.PerceptionKey.WEBCAM_SCENE,
            {
                "name": "webcam.png",
                "mime_type": "image/png",
                "base64_data": BLUE_PNG_BASE64,
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
        self.assertEqual([attachment.name for attachment in attachments], ["screen.png", "webcam.png"])
        self.assertEqual(
            [attachment.base64_data for attachment in attachments],
            [RED_PNG_BASE64, BLUE_PNG_BASE64],
        )

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
            "name": "screen-a.png",
            "mime_type": "image/png",
            "data": RED_PNG_BASE64,
        })
        duplicate = server_module.attachment_from_payload({
            "name": "screen-b.png",
            "mime_type": "image/png",
            "data": RED_PNG_BASE64,
        })
        different = server_module.attachment_from_payload({
            "name": "screen-c.png",
            "mime_type": "image/png",
            "data": BLUE_PNG_BASE64,
        })

        deduped = server_module._dedupe_attachments_by_hash([
            first,
            duplicate,
            different,
        ])

        self.assertEqual(
            [attachment.name for attachment in deduped],
            ["screen-a.png", "screen-c.png"],
        )

    def test_parse_user_message_repairs_prefixed_png_clipboard_payload(self):
        broken_png = b"\xbbK\xe0\x00" + base64.b64decode(RED_PNG_BASE64)
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
            outfit_catalog={"default": "/static/avatar.vrm"},
            current_outfit="default",
        )

        self.assertEqual(
            payload,
            {
                "type": "session_init",
                "server_instance_id": "server-1",
                "session_id": "session-a",
                "gesture_catalog": {"greeting": "/static/animations/Gestures/Greeting.fbx"},
                "outfit_catalog": {"default": "/static/avatar.vrm"},
                "current_outfit": "default",
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
                server_module.WebSocketMessageInbox(ws),
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

    def test_request_tool_approval_rejects_non_boolean_decisions(self):
        for value in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(value=value):
                ws = FakeApprovalWebSocket(approved=value)
                with self.assertLogs("server", level="WARNING"):
                    approved = server_module.asyncio.run(
                        server_module._request_tool_approval(
                            server_module.WebSocketMessageInbox(ws),
                            {"tool": "shell__execute"},
                            connection_id="conn-1",
                            timeout_seconds=1.0,
                        )
                    )
                self.assertFalse(approved)

    def test_request_tool_approval_accepts_literal_false_as_denial(self):
        ws = FakeApprovalWebSocket(approved=False)
        approved = server_module.asyncio.run(
            server_module._request_tool_approval(
                server_module.WebSocketMessageInbox(ws),
                {"tool": "shell__execute"},
                connection_id="conn-1",
                timeout_seconds=1.0,
            )
        )
        self.assertFalse(approved)

    def test_request_tool_approval_preserves_interleaved_messages_in_order(self):
        interleaved_messages = [
            {"text": json.dumps({"type": "user_message", "text": "keep me"})},
            {"text": json.dumps({"type": "set_instant_mode", "enabled": True})},
        ]
        ws = FakeInterleavedApprovalWebSocket(interleaved_messages)
        inbox = server_module.WebSocketMessageInbox(ws)

        approved = server_module.asyncio.run(
            server_module._request_tool_approval(
                inbox,
                {"tool": "shell__execute"},
                connection_id="conn-1",
                timeout_seconds=1.0,
            )
        )

        self.assertTrue(approved)
        self.assertEqual(
            server_module.asyncio.run(inbox.receive()),
            interleaved_messages[0],
        )
        self.assertEqual(
            server_module.asyncio.run(inbox.receive()),
            interleaved_messages[1],
        )

    def test_websocket_inbox_accepts_frames_at_byte_limits(self):
        for message, limits in (
            ({"text": "é"}, {"max_text_bytes": 2}),
            ({"bytes": b"12"}, {"max_binary_bytes": 2}),
        ):
            with self.subTest(message=message):
                ws = FakeFrameWebSocket(message)
                inbox = server_module.WebSocketMessageInbox(ws, **limits)
                self.assertEqual(server_module.asyncio.run(inbox.receive()), message)
                self.assertIsNone(ws.close_payload)

    def test_websocket_inbox_closes_oversized_text_and_binary_frames(self):
        for message, limits, expected_kind in (
            ({"text": "éé"}, {"max_text_bytes": 3}, "Text"),
            ({"bytes": b"123"}, {"max_binary_bytes": 2}, "Binary"),
        ):
            with self.subTest(message=message):
                ws = FakeFrameWebSocket(message)
                inbox = server_module.WebSocketMessageInbox(ws, **limits)
                with self.assertRaisesRegex(
                    server_module.WebSocketMessageTooLarge,
                    f"{expected_kind} frame exceeds",
                ):
                    server_module.asyncio.run(inbox.receive())
                self.assertEqual(ws.close_payload["code"], 1009)
                self.assertIn("byte limit", ws.close_payload["reason"])

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
