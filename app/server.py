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
from app.config import Config

from app.core.orchestrator_factory import build_orchestrator
from app.core.events import AssistantSpeechEvent, AssistantStateEvent
from app.logging import setup_logging
from app.tts.piper_tts import PiperTTS
from app.core.sentence_splitter import split_sentences

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

                            tts_start = time.perf_counter()
                            tts.synthesize(sentence, audio_path)

                            logger.debug(
                                "[%s] TTS complete (%.2f ms)",
                                session_id,
                                (time.perf_counter() - tts_start) * 1000,
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

                            tts.synthesize(text_buffer, audio_path)

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
