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

    async def test_stalled_engine_times_out_quarantines_and_removes_late_audio(self):
        import tempfile
        release = threading.Event()
        finished = threading.Event()
        self.addCleanup(release.set)
        calls = []
        def synthesize(text, path):
            calls.append(text)
            release.wait(3)
            path.write_bytes(b"late audio")
            finished.set()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "late.wav"
            delivery = AudioDelivery(SimpleNamespace(synthesize=synthesize), timeout=0.05)
            delivery.start()
            with self.assertRaises(asyncio.TimeoutError):
                await delivery.synthesize("stalled", path, "session")
            # Let the worker's own deadline expire, not just the caller's deadline.
            await asyncio.sleep(0.08)
            with self.assertRaisesRegex(RuntimeError, "restart required"):
                await delivery.synthesize("next", path, "session")
            await asyncio.wait_for(delivery.close(), 0.5)
            self.assertEqual(calls, ["stalled"])
            release.set()
            for _ in range(100):
                if finished.is_set() and not path.exists():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(finished.is_set())
            self.assertFalse(path.exists())

    async def test_saturated_queue_and_shutdown_are_bounded(self):
        release = threading.Event()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        self.addCleanup(release.set)
        calls = []
        def synthesize(text, path):
            calls.append(text)
            loop.call_soon_threadsafe(started.set)
            release.wait(3)
        delivery = AudioDelivery(SimpleNamespace(synthesize=synthesize), queue_size=1,
                                 timeout=0.2, shutdown_timeout=0.03)
        delivery.start()
        jobs = [asyncio.create_task(delivery.synthesize("first", Path("unused"), "s"))]
        await asyncio.wait_for(started.wait(), 1)
        jobs.extend(asyncio.create_task(delivery.synthesize(text, Path("unused"), "s"))
                    for text in ("queued", "blocked"))
        await asyncio.sleep(0.02)
        await asyncio.wait_for(delivery.close(), 0.5)
        results = await asyncio.wait_for(asyncio.gather(*jobs, return_exceptions=True), 0.5)
        self.assertTrue(all(isinstance(result, Exception) for result in results))
        self.assertEqual(calls, ["first"])
        self.assertTrue(delivery.worker_task.done())
        self.assertTrue(delivery._queue.empty())
        release.set()

    def test_stalled_engine_does_not_hold_process_exit_open(self):
        import subprocess
        import sys
        code = '''
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from app.tts.delivery import AudioDelivery
async def main():
    delivery = AudioDelivery(SimpleNamespace(synthesize=lambda *args: threading.Event().wait()),
                             timeout=0.02, shutdown_timeout=0.02)
    delivery.start()
    try:
        await delivery.synthesize("stalled", Path("unused"), "session")
    except asyncio.TimeoutError:
        pass
    await delivery.close()
asyncio.run(main())
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
