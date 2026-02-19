from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from typing import Iterator, Any
import json
import logging
import uuid
import time
from pathlib import Path
from dataclasses import dataclass
from contextlib import suppress
from app.config import Config

from app.core.orchestrator_factory import build_orchestrator
from app.core.events import AssistantSpeechEvent, AssistantStateEvent
from app.logging import setup_logging
from app.tts.piper_tts import PiperTTS
from app.services.sentence_splitter import split_sentences

setup_logging()
logger = logging.getLogger("server")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure audio directory exists
AUDIO_DIR = Path("static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
logger.debug("Audio directory ready at %s", AUDIO_DIR.resolve())

config = Config()

tts = PiperTTS(
    model_path=Path(config.tts["model_path"]),
    use_cuda=config.tts["use_cuda"],
)

logger.info("Starting FastAPI server")

_SENTINEL = object()
_TTS_STOP = object()
TTS_QUEUE_MAXSIZE = 128


@dataclass
class TTSJob:
    text: str
    output_path: Path
    result_future: asyncio.Future[None]
    session_id: str


def _next_or_sentinel(iterator: Iterator[Any]):
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


async def run_generator(gen: Iterator[Any]):
    """
    Run a blocking generator in an executor and re-yield items async.
    """
    loop = asyncio.get_running_loop()
    iterator = iter(gen)

    logger.debug("Starting generator bridge")

    while True:
        item = await loop.run_in_executor(
            None, _next_or_sentinel, iterator
        )

        if item is _SENTINEL:
            logger.debug("Generator exhausted")
            break

        yield item


async def tts_worker(queue: asyncio.Queue):
    """
    Single TTS worker that serializes synth requests and keeps blocking work
    off the asyncio event loop.
    """
    loop = asyncio.get_running_loop()
    logger.info("TTS worker started")

    while True:
        job = await queue.get()
        try:
            if job is _TTS_STOP:
                logger.info("TTS worker stopping")
                return

            tts_start = time.perf_counter()
            await loop.run_in_executor(
                None,
                tts.synthesize,
                job.text,
                job.output_path,
            )

            logger.debug(
                "[%s] TTS complete (%.2f ms)",
                job.session_id,
                (time.perf_counter() - tts_start) * 1000,
            )

            if not job.result_future.done():
                job.result_future.set_result(None)

        except Exception as exc:
            if isinstance(job, TTSJob) and not job.result_future.done():
                job.result_future.set_exception(exc)
            logger.exception("TTS worker failed to synthesize audio")

        finally:
            queue.task_done()


async def synthesize_async(text: str, output_path: Path, session_id: str):
    queue: asyncio.Queue = app.state.tts_queue
    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[None] = loop.create_future()

    await queue.put(
        TTSJob(
            text=text,
            output_path=output_path,
            result_future=result_future,
            session_id=session_id,
        )
    )
    await result_future


@app.on_event("startup")
async def startup_event():
    app.state.tts_queue = asyncio.Queue(maxsize=TTS_QUEUE_MAXSIZE)
    app.state.tts_worker_task = asyncio.create_task(tts_worker(app.state.tts_queue))
    logger.info("TTS queue initialized (maxsize=%d)", TTS_QUEUE_MAXSIZE)


@app.on_event("shutdown")
async def shutdown_event():
    queue = getattr(app.state, "tts_queue", None)
    worker_task = getattr(app.state, "tts_worker_task", None)

    if queue is not None:
        await queue.put(_TTS_STOP)

    if worker_task is not None:
        with suppress(asyncio.CancelledError):
            await worker_task


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    session_id = uuid.uuid4().hex[:8]
    start_ts = time.perf_counter()

    await ws.accept()
    logger.info("[%s] WebSocket connected", session_id)

    orchestrator = build_orchestrator()
    logger.debug("[%s] Orchestrator created", session_id)

    try:
        while True:
            user_text = await ws.receive_text()

            logger.info(
                "[%s] Received user input (len=%d)",
                session_id,
                len(user_text),
            )
            logger.debug("[%s] User input text: %r", session_id, user_text)

            # Buffer for sentence-based TTS
            text_buffer = ""

            async for event in run_generator(
                orchestrator.handle_user_input(user_text)
            ):
                # --- STATE EVENTS ---
                if isinstance(event, AssistantStateEvent):
                    logger.debug(
                        "[%s] Assistant state -> %s",
                        session_id,
                        event.state,
                    )
                    await ws.send_text(json.dumps({
                        "type": "assistant_state",
                        "state": event.state,
                    }))
                    continue

                # --- SPEECH EVENTS ---
                if isinstance(event, AssistantSpeechEvent):
                    if not event.is_final:
                        text_buffer += event.text

                        await ws.send_text(json.dumps({
                            "type": "assistant_chunk",
                            "content": event.text,
                        }))

                        sentences, text_buffer = split_sentences(text_buffer)

                        for sentence in sentences:
                            audio_id = uuid.uuid4().hex
                            audio_path = AUDIO_DIR / f"{audio_id}.wav"

                            logger.debug(
                                "[%s] TTS synth sentence (%d chars)",
                                session_id,
                                len(sentence),
                            )

                            await synthesize_async(
                                text=sentence,
                                output_path=audio_path,
                                session_id=session_id,
                            )

                            await ws.send_text(json.dumps({
                                "type": "assistant_audio",
                                "url": f"/static/audio/{audio_id}.wav",
                            }))

                    else:
                        if text_buffer.strip():
                            audio_id = uuid.uuid4().hex
                            audio_path = AUDIO_DIR / f"{audio_id}.wav"

                            logger.debug(
                                "[%s] TTS final fragment (%d chars)",
                                session_id,
                                len(text_buffer),
                            )

                            await synthesize_async(
                                text=text_buffer,
                                output_path=audio_path,
                                session_id=session_id,
                            )

                            await ws.send_text(json.dumps({
                                "type": "assistant_audio",
                                "url": f"/static/audio/{audio_id}.wav",
                            }))

                        await ws.send_text(json.dumps({
                            "type": "assistant_end",
                            "content": event.text,
                        }))

                        logger.info(
                            "[%s] Assistant turn completed",
                            session_id,
                        )

    except WebSocketDisconnect:
        logger.info(
            "[%s] WebSocket disconnected (uptime=%.2f s)",
            session_id,
            time.perf_counter() - start_ts,
        )

    except Exception:
        logger.exception("[%s] WebSocket handler crashed", session_id)

    finally:
        logger.debug("[%s] WebSocket cleanup complete", session_id)


@app.get("/")
async def get_index():
    logger.debug("Serving index.html")
    return FileResponse("static/index.html")
