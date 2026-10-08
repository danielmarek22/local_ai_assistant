"""Preserve a reply's speech while bounding each optional synthesis/delivery wait."""
import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from types import TracebackType

logger = logging.getLogger("server")


class SpeechStream:
    def __init__(self, send_speech: Callable[[str], Awaitable[None]], *,
                 fragment_timeout: float = 30.0) -> None:
        if not math.isfinite(fragment_timeout) or fragment_timeout <= 0:
            raise ValueError("Speech fragment timeout must be finite and positive")
        self._send_speech = send_speech
        # Buffer only unsynthesized text for this reply, never waveforms or model
        # instances. Sentence count must not truncate a buffered model response.
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._fragment_timeout = fragment_timeout
        self._task: asyncio.Task[None] | None = None
        self._disabled = False
        self._closed = False

    async def __aenter__(self) -> "SpeechStream":
        return self

    def submit(self, text: str) -> None:
        if not text or self._disabled or self._closed:
            return
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        self._queue.put_nowait(text)

    async def _run(self) -> None:
        try:
            while True:
                text = await self._queue.get()
                try:
                    if text is None:
                        return
                    # A healthy long reply may take more than 30 seconds in total.
                    # Only a stalled fragment should abandon its remaining speech.
                    await asyncio.wait_for(self._send_speech(text), self._fragment_timeout)
                finally:
                    self._queue.task_done()
        except asyncio.TimeoutError:
            logger.warning("Speech fragment timed out; continuing remaining turn with text only")
        except Exception:
            logger.warning("Speech failed; continuing turn with text only", exc_info=True)
        finally:
            self._disabled = True
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc: BaseException | None, tb: TracebackType | None) -> None:
        self._closed = True
        if self._task is None:
            return
        try:
            if exc_type is None:
                if not self._task.done():
                    self._queue.put_nowait(None)
                await self._task
        finally:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
