"""Own queued speech synthesis and its application lifetime."""
import asyncio
import logging
import time
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
    def __init__(self, engine: TTS | None, *, queue_size: int = 128) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
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
        async with self._admission:
            if self._closing or self.worker_task is None or self.worker_task.done():
                raise RuntimeError("Audio delivery is not running")
            future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            await self._queue.put(_SpeechJob(text, output_path, future, session_id))
        await future

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._drain())
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        # Serialize shutdown behind submissions already waiting for queue space.
        async with self._admission:
            if self.worker_task is None:
                return
            await self._queue.put(None)
        await self.worker_task

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        logger.info("TTS worker started")
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    logger.info("TTS worker stopping")
                    return
                if self.engine is None:
                    raise RuntimeError("TTS engine has not been initialized")
                started = time.perf_counter()
                await loop.run_in_executor(None, self.engine.synthesize, job.text, job.output_path)
                logger.debug("[%s] TTS complete (%.2f ms)", job.session_id,
                             (time.perf_counter() - started) * 1000)
                if not job.result_future.done():
                    job.result_future.set_result(None)
            except Exception as exc:
                logger.exception("TTS worker failed to synthesize audio")
                if isinstance(job, _SpeechJob) and not job.result_future.done():
                    # Do not expose the live worker frame to a caller clearing
                    # exception tracebacks. Full backend diagnostics are logged above.
                    job.result_future.set_exception(exc.with_traceback(None))
            finally:
                self._queue.task_done()
