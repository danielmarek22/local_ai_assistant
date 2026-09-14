import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import server
from app.core.events import AssistantSpeechEvent
from app.tts.stream import SpeechStream


class SpeechStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_finishes_before_speech_is_released(self):
        started = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()
        frames = []
        async def synthesize(*args, **kwargs):
            started.set()
            await release.wait()
        async def send(ws, payload):
            frames.append(payload)
            if payload["type"] == "assistant_end":
                completed.set()
        events = iter([AssistantSpeechEvent(text="Hello.", is_final=False),
                       AssistantSpeechEvent(text="Hello.", is_final=True)])
        with patch.object(server, "synthesize_async", synthesize), patch.object(server, "_send_ws_payload", send):
            task = asyncio.create_task(server._forward_orchestrator_events(
                object(), SimpleNamespace(), events, "session", 0, application=SimpleNamespace()))
            try:
                await asyncio.wait_for(completed.wait(), 1)
                await asyncio.wait_for(started.wait(), 1)
                self.assertEqual([frame["type"] for frame in frames], ["assistant_chunk", "assistant_end"])
                self.assertFalse(task.done())
                release.set()
                await asyncio.wait_for(task, 1)
                self.assertEqual(frames[-1]["type"], "assistant_audio")
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_backlog_saturation_drops_speech_without_blocking(self):
        send = AsyncMock(side_effect=lambda text: asyncio.sleep(10))
        async with SpeechStream(send, queue_size=1, drain_timeout=0.02) as speech:
            speech.submit("one")
            speech.submit("two")
            speech.submit("three")
        send.assert_not_awaited()

    async def test_stalled_speech_drain_is_bounded_and_cancelled(self):
        stopped = asyncio.Event()
        async def send(text):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        async def turn():
            async with SpeechStream(send, drain_timeout=0.02) as speech:
                speech.submit("one")
        await asyncio.wait_for(turn(), 0.5)
        self.assertTrue(stopped.is_set())

    async def test_failure_abandons_remaining_speech(self):
        send = AsyncMock(side_effect=RuntimeError("Unavailable"))
        async with SpeechStream(send, drain_timeout=0.02) as speech:
            speech.submit("one")
            speech.submit("two")
        send.assert_awaited_once_with("one")

    async def test_autonomous_text_finishes_before_optional_speech(self):
        frames = []
        async def broadcast(session, frame):
            frames.append(frame)
        async def synthesize(*args):
            self.assertEqual([f["type"] for f in frames], ["assistant_chunk", "assistant_end"])
        app = SimpleNamespace(state=SimpleNamespace(
            connection_hub=SimpleNamespace(broadcast=broadcast, has_session=lambda session: True),
            audio_delivery=object()))
        with patch.object(server, "synthesize_async", synthesize):
            await server._autonomy_notification_sink(
                "session", {"message": "Hello", "delivery": "speech"}, "turn", application=app)
        self.assertEqual(frames[-1]["type"], "assistant_audio")
        self.assertEqual(frames[-1]["turn_id"], "turn")
