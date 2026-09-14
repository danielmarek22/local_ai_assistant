"""Bound optional speech without delaying text generation or completion frames."""
import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from types import TracebackType

logger = logging.getLogger("server")


class SpeechStream:
    def __init__(self, send_speech: Callable[[str], Awaitable[None]], *,
                 queue_size: int = 8, drain_timeout: float = 30.0) -> None:
        if queue_size < 1 or not math.isfinite(drain_timeout) or drain_timeout <= 0:
            raise ValueError("Speech queue size and drain timeout must be positive")
        self._send_speech = send_speech
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
        self._drain_timeout = drain_timeout
        self._task: asyncio.Task[None] | None = None
        self._disabled = False

    async def __aenter__(self) -> "SpeechStream":
        return self

    def submit(self, text: str) -> None:
        if not text or self._disabled:
            return
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning("Speech backlog full; continuing turn with text only")
            self._disabled = True
            self._task.cancel()

    async def _run(self) -> None:
        try:
            while True:
                text = await self._queue.get()
                try:
                    await self._send_speech(text)
                finally:
                    self._queue.task_done()
        except Exception:
            self._disabled = True
            logger.warning("Speech failed; continuing turn with text only", exc_info=True)

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc: BaseException | None, tb: TracebackType | None) -> None:
        if self._task is None:
            return
        try:
            if exc_type is None and not self._disabled:
                # A failed consumer must not leave join waiting on queued items.
                joined = asyncio.create_task(self._queue.join())
                try:
                    await asyncio.wait(
                        (joined, self._task), timeout=self._drain_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    joined.cancel()
                    await asyncio.gather(joined, return_exceptions=True)
        finally:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
