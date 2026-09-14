import asyncio
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.tts.delivery import AudioDelivery


class AudioDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_drains_admitted_jobs_including_waiter_for_queue_space(self):
        started = asyncio.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        loop = asyncio.get_running_loop()
        calls = []
        def synthesize(text, path):
            calls.append(text)
            if text == "first":
                loop.call_soon_threadsafe(started.set)
                if not release.wait(3):
                    raise TimeoutError("Test release missing")
        delivery = AudioDelivery(SimpleNamespace(synthesize=synthesize), queue_size=1)
        delivery.start()
        jobs = [asyncio.create_task(delivery.synthesize("first", Path("unused"), "session"))]
        await asyncio.wait_for(started.wait(), 1)
        for text in ("second", "third"):
            jobs.append(asyncio.create_task(delivery.synthesize(text, Path("unused"), "session")))
            await asyncio.sleep(0)
        closing = asyncio.create_task(delivery.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        release.set()
        await asyncio.wait_for(asyncio.gather(*jobs, closing), 2)
        self.assertEqual(calls, ["first", "second", "third"])
        self.assertTrue(delivery.worker_task.done())
        await delivery.close()
        with self.assertRaisesRegex(RuntimeError, "not running"):
            await delivery.synthesize("late", Path("unused"), "session")

    async def test_backend_failure_reaches_caller_and_next_job_still_runs(self):
        calls = []
        def synthesize(text, path):
            calls.append(text)
            if text == "bad":
                raise ValueError("Backend failed")
        delivery = AudioDelivery(SimpleNamespace(synthesize=synthesize))
        delivery.start()
        try:
            with self.assertLogs("server", level="ERROR"):
                with self.assertRaisesRegex(ValueError, "Backend failed"):
                    await delivery.synthesize("bad", Path("unused"), "session")
            await delivery.synthesize("good", Path("unused"), "session")
            self.assertEqual(calls, ["bad", "good"])
        finally:
            await delivery.close()

    async def test_cancelled_caller_does_not_break_worker(self):
        delivery = AudioDelivery(SimpleNamespace(synthesize=lambda text, path: None))
        delivery.start()
        job = asyncio.create_task(delivery.synthesize("cancel", Path("unused"), "session"))
        await asyncio.sleep(0)
        job.cancel()
        await asyncio.gather(job, return_exceptions=True)
        await delivery.synthesize("next", Path("unused"), "session")
        await delivery.close()
        self.assertTrue(delivery.worker_task.done())

    async def test_unstarted_delivery_can_close_and_cannot_restart(self):
        delivery = AudioDelivery(None)
        await delivery.close()
        await delivery.close()
        with self.assertRaises(RuntimeError):
            delivery.start()

    async def test_cancelled_shutdown_caller_does_not_abandon_drain(self):
        started = asyncio.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        loop = asyncio.get_running_loop()
        def synthesize(text, path):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(3):
                raise TimeoutError("Test release missing")
        delivery = AudioDelivery(SimpleNamespace(synthesize=synthesize))
        delivery.start()
        job = asyncio.create_task(delivery.synthesize("speech", Path("unused"), "session"))
        await asyncio.wait_for(started.wait(), 1)
        closing = asyncio.create_task(delivery.close())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.gather(closing, return_exceptions=True)
        release.set()
        await asyncio.wait_for(asyncio.gather(job, delivery.close()), 2)
        self.assertTrue(delivery.worker_task.done())
