"""Own queued speech synthesis and its application lifetime."""
import asyncio
import logging
import time
import math
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from app.tts.base import TTS

logger = logging.getLogger("server")


@dataclass
class _SpeechJob:
    text: str
    output_path: Path
    result_future: asyncio.Future[None]
    session_id: str


class AudioDelivery:
    def __init__(self, engine: TTS | None, *, queue_size: int = 128,
                 timeout: float = 30.0, shutdown_timeout: float = 5.0) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        if not all(math.isfinite(value) and value > 0 for value in (timeout, shutdown_timeout)):
            raise ValueError("Audio timeouts must be finite and positive")
        self.timeout = timeout
        self.shutdown_timeout = shutdown_timeout
        self._unavailable = False
        self.engine = engine
        self._queue: asyncio.Queue[_SpeechJob | None] = asyncio.Queue(maxsize=queue_size)
        self._admission = asyncio.Lock()
        self.worker_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False

    def start(self) -> None:
        if self._closing:
            raise RuntimeError("Audio delivery is closing")
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._run())

    async def synthesize(self, text: str, output_path: Path, session_id: str) -> None:
        await asyncio.wait_for(self._submit(text, output_path, session_id), self.timeout)

    async def _submit(self, text: str, output_path: Path, session_id: str) -> None:
        async with self._admission:
            if self._unavailable:
                raise RuntimeError("TTS engine timed out; restart required")
            if self._closing or self.worker_task is None or self.worker_task.done():
                raise RuntimeError("Audio delivery is not running")
            future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            await self._queue.put(_SpeechJob(text, output_path, future, session_id))
            if self.worker_task.done():
                # Shutdown may have released a producer waiting for queue space.
                self._discard_queued()
        try:
            await future
        finally:
            future.cancel()

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._drain())
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        try:
            await asyncio.wait_for(self._finish(), self.shutdown_timeout)
        except asyncio.TimeoutError:
            logger.warning("TTS shutdown deadline reached; abandoning speech")
        finally:
            if self.worker_task is not None:
                self.worker_task.cancel()
                await asyncio.gather(self.worker_task, return_exceptions=True)
            self._discard_queued()

    def _discard_queued(self) -> None:
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if job is not None and not job.result_future.done():
                job.result_future.set_exception(RuntimeError("Audio delivery closed"))
            self._queue.task_done()

    async def _finish(self) -> None:
        async with self._admission:
            if self.worker_task is None:
                return
            await self._queue.put(None)
        await asyncio.shield(self.worker_task)

    async def _synthesize(self, job: _SpeechJob) -> None:
        engine = self.engine
        if engine is None:
            raise RuntimeError("TTS engine has not been initialized")
        result: Future[None] = Future()
        abandoned = threading.Event()

        def cleanup() -> None:
            try:
                job.output_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove abandoned audio %s", job.output_path)

        def run() -> None:
            if not result.set_running_or_notify_cancel():
                return
            try:
                engine.synthesize(job.text, job.output_path)
            except BaseException as exc:
                result.set_exception(exc)
            else:
                result.set_result(None)
            finally:
                if abandoned.is_set():
                    cleanup()

        # Local engines cannot be forcibly stopped. A daemon avoids tying process
        # exit to an unresponsive backend; quarantine prevents overlapping calls.
        threading.Thread(target=run, name="astra-speech", daemon=True).start()
        try:
            await asyncio.wait_for(asyncio.wrap_future(result), self.timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._unavailable = True
            abandoned.set()
            cleanup()
            raise

    async def _run(self) -> None:
        logger.info("TTS worker started")
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    logger.info("TTS worker stopping")
                    return
                if job.result_future.cancelled():
                    continue
                if self._unavailable:
                    raise RuntimeError("TTS engine timed out; restart required")
                started = time.perf_counter()
                await self._synthesize(job)
                logger.debug("[%s] TTS complete (%.2f ms)", job.session_id,
                             (time.perf_counter() - started) * 1000)
                if job.result_future.cancelled():
                    job.output_path.unlink(missing_ok=True)
                elif not job.result_future.done():
                    job.result_future.set_result(None)
            except asyncio.CancelledError:
                if isinstance(job, _SpeechJob) and not job.result_future.done():
                    job.result_future.set_exception(RuntimeError("Audio delivery closed"))
                raise
            except Exception as exc:
                logger.exception("TTS worker failed to synthesize audio")
                if isinstance(job, _SpeechJob) and not job.result_future.done():
                    # Do not expose the live worker frame to a caller clearing
                    # exception tracebacks. Full backend diagnostics are logged above.
                    job.result_future.set_exception(exc.with_traceback(None))
            finally:
                self._queue.task_done()
