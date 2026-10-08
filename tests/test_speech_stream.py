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

    async def test_sentence_burst_is_preserved_without_blocking_submit(self):
        send = AsyncMock()
        sentences = [f"Sentence {index}." for index in range(25)]
        async with SpeechStream(send, fragment_timeout=0.2) as speech:
            for sentence in sentences:
                speech.submit(sentence)
            send.assert_not_awaited()
        self.assertEqual([call.args[0] for call in send.await_args_list], sentences)

    async def test_backlog_during_synthesis_preserves_all_audio_in_order(self):
        started = asyncio.Event()
        release = asyncio.Event()
        generated = []
        delivered = []
        cancelled = asyncio.Event()

        async def send(text):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            generated.append(text)
            delivered.append(text)

        async with SpeechStream(send, fragment_timeout=0.2) as speech:
            speech.submit("in flight")
            await asyncio.wait_for(started.wait(), 0.2)
            speech.submit("queued")
            speech.submit("overflow")
            speech.submit("later")
            await asyncio.sleep(0)
            self.assertFalse(cancelled.is_set())
            release.set()
        self.assertEqual(generated, ["in flight", "queued", "overflow", "later"])
        self.assertEqual(delivered, generated)

    async def test_buffered_reply_publishes_audio_for_every_sentence_and_final_remainder(self):
        generated = []
        frames = []

        async def synthesize(text, *args):
            generated.append(text)

        async def send(ws, payload):
            frames.append(payload)

        sentences = [f"Sentence {index}." for index in range(20)] + ["Final words without punctuation"]
        text = " ".join(sentences)
        events = iter([AssistantSpeechEvent(text=text, is_final=False),
                       AssistantSpeechEvent(text=text, is_final=True)])
        with patch.object(server, "synthesize_async", synthesize), \
                patch.object(server, "_send_ws_payload", send):
            await asyncio.wait_for(server._forward_orchestrator_events(
                object(), SimpleNamespace(), events, "session", 0,
                application=SimpleNamespace()), 0.5)
        self.assertEqual(generated, sentences)
        self.assertEqual([frame["type"] for frame in frames if frame["type"] != "assistant_audio"],
                         ["assistant_chunk", "assistant_end"])
        audio_frames = [frame for frame in frames if frame["type"] == "assistant_audio"]
        self.assertEqual(len(audio_frames), len(sentences))
        self.assertTrue(all(frame["url"].startswith("/static/audio/") for frame in audio_frames))
        self.assertEqual(len({frame["url"] for frame in audio_frames}), len(sentences))

    async def test_stalled_first_fragment_abandons_backlog_within_its_deadline(self):
        stopped = asyncio.Event()

        async def send(text):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def turn():
            async with SpeechStream(send, fragment_timeout=0.02) as speech:
                speech.submit("one")
                speech.submit("overflow")

        await asyncio.wait_for(turn(), 0.5)
        self.assertTrue(stopped.is_set())

    async def test_stalled_speech_is_bounded_and_cancelled(self):
        stopped = asyncio.Event()
        async def send(text):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        async def turn():
            async with SpeechStream(send, fragment_timeout=0.02) as speech:
                speech.submit("one")
        await asyncio.wait_for(turn(), 0.5)
        self.assertTrue(stopped.is_set())

    async def test_failure_abandons_remaining_speech(self):
        send = AsyncMock(side_effect=RuntimeError("Unavailable"))
        async with SpeechStream(send, fragment_timeout=0.02) as speech:
            speech.submit("one")
            speech.submit("two")
        send.assert_awaited_once_with("one")

    async def test_healthy_reply_can_exceed_one_fragment_deadline_in_total(self):
        delivered = []
        active = 0
        max_active = 0

        async def send(text):
            nonlocal active, max_active
            active += 1
            max_active = max(active, max_active)
            await asyncio.sleep(0.03)
            delivered.append(text)
            active -= 1

        async def turn():
            async with SpeechStream(send, fragment_timeout=0.1) as speech:
                for index in range(6):
                    speech.submit(str(index))

        await asyncio.wait_for(turn(), 1)
        self.assertEqual(delivered, [str(index) for index in range(6)])
        self.assertEqual(max_active, 1)

    async def test_generation_failure_cancels_pending_speech(self):
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def send(text):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            async with SpeechStream(send) as speech:
                speech.submit("one")
                speech.submit("two")
                await asyncio.wait_for(started.wait(), 0.5)
                raise RuntimeError("generation failed")
        self.assertTrue(stopped.is_set())

    def test_invalid_fragment_deadlines_are_rejected(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                SpeechStream(AsyncMock(), fragment_timeout=timeout)

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
