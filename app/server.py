from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from typing import Iterator, Any
import json
import logging
import uuid
from pathlib import Path

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

tts = PiperTTS(
    model_path=Path("models/piper/en_US-amy-medium.onnx"),
    use_cuda=False,
)

logger.info("Starting FastAPI server")


_SENTINEL = object()


def _next_or_sentinel(iterator: Iterator[Any]):
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


async def run_generator(gen: Iterator[Any]):
    loop = asyncio.get_running_loop()
    iterator = iter(gen)

    while True:
        item = await loop.run_in_executor(
            None, _next_or_sentinel, iterator
        )

        if item is _SENTINEL:
            break

        yield item


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected")

    orchestrator = build_orchestrator()

    try:
        while True:
            user_text = await ws.receive_text()
            logger.info("Received user input via WebSocket")

            # Buffer for sentence-based TTS
            text_buffer = ""

            async for event in run_generator(
                orchestrator.handle_user_input(user_text)
            ):
                # --- STATE EVENTS ---
                if isinstance(event, AssistantStateEvent):
                    await ws.send_text(json.dumps({
                        "type": "assistant_state",
                        "state": event.state,
                    }))
                    continue

                # --- SPEECH EVENTS ---
                if isinstance(event, AssistantSpeechEvent):
                    # Accumulate streamed text
                    if not event.is_final:
                        text_buffer += event.text

                        # Forward text chunk to frontend
                        await ws.send_text(json.dumps({
                            "type": "assistant_chunk",
                            "content": event.text,
                        }))

                        # Sentence detection
                        sentences, text_buffer = split_sentences(text_buffer)

                        for sentence in sentences:
                            audio_id = uuid.uuid4().hex
                            audio_path = AUDIO_DIR / f"{audio_id}.wav"

                            logger.debug("Synthesizing sentence: %s", sentence)
                            tts.synthesize(sentence, audio_path)

                            await ws.send_text(json.dumps({
                                "type": "assistant_audio",
                                "url": f"/static/audio/{audio_id}.wav",
                            }))

                    else:
                        # Final event: synthesize leftover text if any
                        if text_buffer.strip():
                            audio_id = uuid.uuid4().hex
                            audio_path = AUDIO_DIR / f"{audio_id}.wav"

                            logger.debug(
                                "Synthesizing final fragment: %s",
                                text_buffer
                            )
                            tts.synthesize(text_buffer, audio_path)

                            await ws.send_text(json.dumps({
                                "type": "assistant_audio",
                                "url": f"/static/audio/{audio_id}.wav",
                            }))

                        # Signal end of assistant turn
                        await ws.send_text(json.dumps({
                            "type": "assistant_end",
                            "content": event.text,
                        }))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


@app.get("/")
async def get_index():
    return FileResponse("static/index.html")
