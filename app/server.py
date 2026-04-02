from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import asyncio
from typing import Iterator, Any
import json
import logging
import re
import uuid
import time
from pathlib import Path
from dataclasses import dataclass
from contextlib import suppress
from app.config import Config

from app.core.assistant_state import AssistantState
from app.core.orchestrator_factory import build_orchestrator
from app.core.events import (
    AssistantSpeechEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
)
from app.logging import setup_logging_from_config
from app.perception.state import ImageAttachment
from app.tts.factory import build_tts_engine
from app.services.sentence_splitter import split_sentences

config = Config()
setup_logging_from_config(config.logging)
logger = logging.getLogger("server")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure audio directory exists
AUDIO_DIR = Path("static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
logger.debug("Audio directory ready at %s", AUDIO_DIR.resolve())

tts = build_tts_engine(config.tts)

logger.info("Starting FastAPI server")

_SENTINEL = object()
_TTS_STOP = object()
TTS_QUEUE_MAXSIZE = 128
_FENCED_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_MARKER_RE = re.compile(r"[*_~`>#]")


def _prepare_tts_text(text: str) -> str:
    """
    Lightweight markdown skipping for TTS:
    - drop fenced code blocks
    - keep link labels, drop URLs
    - remove common markdown marker chars
    """
    if not text:
        return ""

    cleaned = _FENCED_CODE_BLOCK_RE.sub(" ", text)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_MARKER_RE.sub("", cleaned)
    return cleaned.strip()


def resolve_session_id(
    session_mode: str,
    requested_session_id: str | None,
    known_server_instance_id: str | None,
    server_instance_id: str,
) -> str:
    if session_mode == "open" and requested_session_id:
        return requested_session_id

    if (
        session_mode == "resume"
        and requested_session_id
        and known_server_instance_id == server_instance_id
    ):
        return requested_session_id

    return uuid.uuid4().hex[:8]


def parse_user_message(raw_text: str) -> tuple[str, bool | None, list[ImageAttachment]]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text, None, []

    if not isinstance(payload, dict):
        return raw_text, None, []

    if payload.get("type") != "user_message":
        return raw_text, None, []

    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("User message payload is missing text")

    attachments_payload = payload.get("attachments")
    if attachments_payload is None:
        attachments: list[ImageAttachment] = []
    elif isinstance(attachments_payload, list):
        attachments = [ImageAttachment.from_payload(item) for item in attachments_payload]
    else:
        raise ValueError("User message attachments must be a list")

    if not text.strip() and not attachments:
        raise ValueError("User message must include text or at least one image attachment")

    reasoning = payload.get("reasoning")
    if reasoning is None:
        reasoning_override = None
    elif isinstance(reasoning, bool):
        reasoning_override = reasoning
    else:
        raise ValueError("User message reasoning flag must be boolean")

    return text, reasoning_override, attachments


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


async def _send_ws_payload(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload))


async def _flush_pending_chunks(ws: WebSocket, pending_chunks: list[str]) -> None:
    for chunk in pending_chunks:
        await _send_ws_payload(ws, {
            "type": "assistant_chunk",
            "content": chunk,
        })
    pending_chunks.clear()


def _should_forward_state(state: str) -> bool:
    return state != AssistantState.RESPONDING


@app.on_event("startup")
async def startup_event():
    app.state.server_instance_id = uuid.uuid4().hex[:8]
    app.state.orchestrator = build_orchestrator()
    logger.info(
        "Orchestrator initialized at startup (server_instance_id=%s)",
        app.state.server_instance_id,
    )

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


@app.get("/api/sessions")
async def list_sessions():
    history_store = app.state.orchestrator.history
    rows = history_store.list_sessions()
    sessions = [
        {
            "session_id": row["session_id"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "message_count": row["message_count"],
            "preview": row["preview"],
        }
        for row in rows
    ]
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    history_store = app.state.orchestrator.history
    rows = history_store.get_all(session_id)

    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")

    summary = app.state.orchestrator.summary_store.get(session_id)
    summary_text = summary[0] if summary else None
    return {
        "session_id": session_id,
        "summary": summary_text,
        "messages": [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "attachments": [attachment.to_api_payload() for attachment in row.get("attachments", [])],
            }
            for row in rows
        ],
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    orchestrator = app.state.orchestrator
    deleted_count = orchestrator.history.delete_session(session_id)
    orchestrator.summary_store.delete(session_id)

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"deleted": True, "session_id": session_id}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    connection_id = uuid.uuid4().hex[:8]
    start_ts = time.perf_counter()
    server_instance_id = app.state.server_instance_id
    session_mode = ws.query_params.get("session_mode", "new")
    requested_session_id = ws.query_params.get("session_id")
    known_server_instance_id = ws.query_params.get("server_instance_id")

    session_id = resolve_session_id(
        session_mode=session_mode,
        requested_session_id=requested_session_id,
        known_server_instance_id=known_server_instance_id,
        server_instance_id=server_instance_id,
    )

    await ws.accept()
    logger.info(
        "[%s] WebSocket connected (conversation_session=%s, mode=%s)",
        connection_id,
        session_id,
        session_mode,
    )
    await ws.send_text(json.dumps({
        "type": "session_init",
        "server_instance_id": server_instance_id,
        "session_id": session_id,
    }))

    orchestrator = app.state.orchestrator
    logger.debug("[%s] Reusing startup orchestrator", connection_id)

    try:
        while True:
            raw_message = await ws.receive_text()
            user_text, reasoning_override, attachments = parse_user_message(raw_message)

            logger.info(
                "[%s] Received user input (len=%d, images=%d, reasoning_override=%r)",
                connection_id,
                len(user_text),
                len(attachments),
                reasoning_override,
            )
            logger.debug("[%s] User input text: %r", connection_id, user_text)

            # Buffer for sentence-based TTS
            text_buffer = ""
            pending_chunks: list[str] = []
            text_released = False
            tts_enabled = True

            async for event in run_generator(
                orchestrator.handle_user_input(
                    session_id,
                    user_text,
                    think_override=reasoning_override,
                    attachments=attachments,
                )
            ):
                # --- STATE EVENTS ---
                if isinstance(event, AssistantStateEvent):
                    if not _should_forward_state(event.state):
                        logger.debug(
                            "[%s] Holding assistant state at thinking until audio is ready",
                            connection_id,
                        )
                        continue

                    logger.debug(
                        "[%s] Assistant state -> %s",
                        connection_id,
                        event.state,
                    )
                    await _send_ws_payload(ws, {
                        "type": "assistant_state",
                        "state": event.state,
                    })
                    continue

                if isinstance(event, AvatarExpressionEvent):
                    logger.debug(
                        "[%s] Avatar expression -> %s",
                        connection_id,
                        event.expression,
                    )
                    await _send_ws_payload(ws, {
                        "type": "assistant_expression",
                        "expression": event.expression,
                    })
                    continue

                # --- SPEECH EVENTS ---
                if isinstance(event, AssistantSpeechEvent):
                    if not event.is_final:
                        text_buffer += event.text

                        if text_released:
                            await _send_ws_payload(ws, {
                                "type": "assistant_chunk",
                                "content": event.text,
                            })
                        else:
                            pending_chunks.append(event.text)

                        sentences, text_buffer = split_sentences(text_buffer)

                        if not tts_enabled:
                            continue

                        for sentence in sentences:
                            tts_text = _prepare_tts_text(sentence)
                            if not tts_text:
                                continue

                            audio_id = uuid.uuid4().hex
                            audio_path = AUDIO_DIR / f"{audio_id}.wav"

                            logger.debug(
                                "[%s] TTS synth sentence (%d chars)",
                                connection_id,
                                len(tts_text),
                            )

                            try:
                                await synthesize_async(
                                    text=tts_text,
                                    output_path=audio_path,
                                    session_id=connection_id,
                                )
                            except Exception:
                                tts_enabled = False
                                logger.warning(
                                    "[%s] TTS failed mid-turn; falling back to text-only streaming",
                                    connection_id,
                                )
                                if not text_released:
                                    await _flush_pending_chunks(ws, pending_chunks)
                                    text_released = True
                                break

                            await _send_ws_payload(ws, {
                                "type": "assistant_audio",
                                "url": f"/static/audio/{audio_id}.wav",
                            })

                            if not text_released:
                                await _flush_pending_chunks(ws, pending_chunks)
                                text_released = True

                    else:
                        if tts_enabled:
                            tts_text = _prepare_tts_text(text_buffer)
                            if tts_text:
                                audio_id = uuid.uuid4().hex
                                audio_path = AUDIO_DIR / f"{audio_id}.wav"

                                logger.debug(
                                    "[%s] TTS final fragment (%d chars)",
                                    connection_id,
                                    len(tts_text),
                                )

                                try:
                                    await synthesize_async(
                                        text=tts_text,
                                        output_path=audio_path,
                                        session_id=connection_id,
                                    )
                                except Exception:
                                    tts_enabled = False
                                    logger.warning(
                                        "[%s] TTS failed for final fragment; sending text without audio",
                                        connection_id,
                                    )
                                else:
                                    await _send_ws_payload(ws, {
                                        "type": "assistant_audio",
                                        "url": f"/static/audio/{audio_id}.wav",
                                    })

                        if not text_released:
                            await _flush_pending_chunks(ws, pending_chunks)
                            text_released = True

                        await _send_ws_payload(ws, {
                            "type": "assistant_end",
                            "content": event.text,
                        })

                        logger.info(
                            "[%s] Assistant turn completed",
                            connection_id,
                        )

    except WebSocketDisconnect:
        logger.info(
            "[%s] WebSocket disconnected (uptime=%.2f s)",
            connection_id,
            time.perf_counter() - start_ts,
        )

    except Exception:
        logger.exception("[%s] WebSocket handler crashed", connection_id)

    finally:
        logger.debug("[%s] WebSocket cleanup complete", connection_id)


@app.get("/")
async def get_index():
    logger.debug("Serving index.html")
    return FileResponse("static/index.html")
