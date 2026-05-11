from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import asyncio
from typing import Iterator, Any
import hashlib
import json
import logging
import re
import uuid
import time
import emoji
from pathlib import Path
from dataclasses import dataclass
from contextlib import suppress
from pydantic import BaseModel, Field
from app.config import Config

from app.core.assistant_state import AssistantState
from app.core.orchestrator_factory import build_orchestrator
from app.core.turn_input import InputModality
from app.core.events import (
    AssistantSpeechEvent,
    AssistantThinkingEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
    AvatarAnimationEvent,
)
from app.logging import setup_logging_from_config
from app.perception.attachments import Attachment, attachment_from_payload
from app.perception.keys import PerceptionKey
from app.tts.factory import build_tts_engine
from app.stt.factory import build_stt_engine
from app.services.sentence_splitter import split_sentences
from app.services.memory_reflector import MemoryReflector
from app.services.vision_watchdog import VisionWatchdog
from app.core.thinking_filter import ThinkingBlockFilter, strip_complete_thinking_blocks

config = Config()
setup_logging_from_config(config.logging)
logger = logging.getLogger("server")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure audio directory exists
AUDIO_DIR = Path("static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
logger.debug("Audio directory ready at %s", AUDIO_DIR.resolve())

tts = None

logger.info("Starting FastAPI server")

_SENTINEL = object()
_TTS_STOP = object()
TTS_QUEUE_MAXSIZE = 128
VISION_CONTEXT_MAX_AGE_SECONDS = 2.0  # Tightened from 5.0s: fallback only for stale frames
_FENCED_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_MARKDOWN_REFERENCE_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$", re.MULTILINE)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_REFERENCE_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_MARKDOWN_AUTOLINK_RE = re.compile(r"<https?://[^>]+>")
_MARKDOWN_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MARKDOWN_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MARKDOWN_UNORDERED_LIST_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_MARKDOWN_ORDERED_LIST_RE = re.compile(r"^\s{0,3}\d+\.\s+", re.MULTILINE)
_MARKDOWN_HRULE_RE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$", re.MULTILINE)
_MARKDOWN_BOLD_RE = re.compile(r"(\*\*|__)(.*?)\1")
_MARKDOWN_ITALIC_RE = re.compile(r"(?<!\w)(\*|_)(.+?)\1(?!\w)")
_MARKDOWN_STRIKE_RE = re.compile(r"~~(.+?)~~")
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>~|])")
_WHITESPACE_RE = re.compile(r"\s+")


def _prepare_tts_text(text: str) -> str:
    """
    Convert markdown-ish text into plain text suitable for TTS:
    - drop fenced code blocks
    - keep link/image labels, drop URLs
    - remove block markers (headings, lists, quotes, rulers)
    - unwrap inline emphasis/code markers
    """
    if not text:
        return ""

    escaped_markers: dict[str, str] = {}

    def _protect_escaped(match: re.Match[str]) -> str:
        token = f"TTSESCAPED{len(escaped_markers)}TOKEN"
        escaped_markers[token] = match.group(1)
        return token

    cleaned = _MARKDOWN_ESCAPE_RE.sub(_protect_escaped, text)
    cleaned = strip_complete_thinking_blocks(cleaned)
    cleaned = _FENCED_CODE_BLOCK_RE.sub(" ", cleaned)
    cleaned = _MARKDOWN_REFERENCE_DEF_RE.sub(" ", cleaned)
    cleaned = _MARKDOWN_IMAGE_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_REFERENCE_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_AUTOLINK_RE.sub(" ", cleaned)
    cleaned = _MARKDOWN_INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_HEADING_RE.sub("", cleaned)
    cleaned = _MARKDOWN_BLOCKQUOTE_RE.sub("", cleaned)
    cleaned = _MARKDOWN_UNORDERED_LIST_RE.sub("", cleaned)
    cleaned = _MARKDOWN_ORDERED_LIST_RE.sub("", cleaned)
    cleaned = _MARKDOWN_HRULE_RE.sub(" ", cleaned)
    cleaned = emoji.replace_emoji(cleaned, replace=" ")

    # Run a few passes so nested emphasis is progressively unwrapped.
    for _ in range(3):
        cleaned = _MARKDOWN_BOLD_RE.sub(r"\2", cleaned)
        cleaned = _MARKDOWN_ITALIC_RE.sub(r"\2", cleaned)
        cleaned = _MARKDOWN_STRIKE_RE.sub(r"\1", cleaned)

    for token, marker in escaped_markers.items():
        cleaned = cleaned.replace(token, marker)

    return _WHITESPACE_RE.sub(" ", cleaned).strip()


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


def parse_user_message(raw_text: str) -> tuple[str, bool | None, list[Attachment]]:
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
        attachments: list[Attachment] = []
    elif isinstance(attachments_payload, list):
        attachments = [attachment_from_payload(item) for item in attachments_payload]
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


class ReflectRequest(BaseModel):
    days_old: int = Field(default=14, ge=0)


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

            if tts is None:
                raise RuntimeError("TTS engine has not been initialized")

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


async def _send_turn_error(ws: WebSocket, message: str) -> None:
    with suppress(Exception):
        await _send_ws_payload(ws, {
            "type": "assistant_state",
            "state": AssistantState.IDLE,
        })

    with suppress(Exception):
        await _send_ws_payload(ws, {
            "type": "assistant_end",
            "content": message,
        })


def _build_session_init_payload(
    server_instance_id: str,
    session_id: str,
    gesture_catalog: dict[str, str] | None = None,
) -> dict:
    return {
        "type": "session_init",
        "server_instance_id": server_instance_id,
        "session_id": session_id,
        "gesture_catalog": dict(gesture_catalog or {}),
    }


def _build_attachment_drop_notice_payload(
    orchestrator,
    original_attachment_count: int,
) -> dict | None:
    if original_attachment_count <= 0:
        return None

    llm = getattr(orchestrator, "llm", None)
    if llm is None:
        return None

    if not getattr(llm, "last_stream_dropped_current_images", False):
        return None

    dropped_count = getattr(llm, "last_stream_dropped_current_images_count", 0)
    if not isinstance(dropped_count, int) or dropped_count <= 0:
        dropped_count = original_attachment_count

    dropped_count = min(dropped_count, original_attachment_count)
    noun = "image" if dropped_count == 1 else "images"
    verb = "was" if dropped_count == 1 else "were"

    return {
        "type": "user_notice",
        "scope": "last_user_message",
        "tone": "warning",
        "message": f"Attached {noun} {verb} not sent to the model for this message.",
    }


def _should_forward_state(state: str) -> bool:
    return state != AssistantState.RESPONDING


def _append_recent_vision_attachments(
    orchestrator,
    attachments: list[Attachment],
    *,
    max_age_seconds: float = VISION_CONTEXT_MAX_AGE_SECONDS,
) -> int:
    # Only inject fallback frames if frontend didn't bundle fresh ones.
    # This ensures synchronized frames from the user message take precedence
    # and prevents mixing fresh bundled frames with stale background polling frames.
    if attachments:
        return 0

    appended_count = 0
    known_hashes = {
        attachment.sha256
        for attachment in attachments
        if getattr(attachment, "sha256", None)
    }

    for key in (PerceptionKey.SCREEN_SCENE, PerceptionKey.WEBCAM_SCENE):
        entry = orchestrator.perception.get(key)
        if entry is None or entry.age > max_age_seconds:
            continue

        payload = entry.value
        if not isinstance(payload, dict):
            continue

        try:
            attachment = attachment_from_payload(payload)
        except ValueError as exc:
            logger.debug("Skipping recent %s perception attachment: %s", key.value, exc)
            continue

        if attachment.sha256 and attachment.sha256 in known_hashes:
            continue

        attachments.append(attachment)
        appended_count += 1
        if attachment.sha256:
            known_hashes.add(attachment.sha256)

    return appended_count


async def _stream_orchestrator_events(
    ws: WebSocket,
    orchestrator,
    event_iterator: Iterator[Any],
    connection_id: str,
    original_attachment_count: int,
    state_tracker: dict[str, str],
) -> None:
    text_buffer = ""
    thinking_filter = ThinkingBlockFilter()
    pending_chunks: list[str] = []
    text_released = False
    tts_enabled = True
    image_notice_sent = False

    async for event in run_generator(event_iterator):
        if not image_notice_sent:
            notice_payload = _build_attachment_drop_notice_payload(
                orchestrator=orchestrator,
                original_attachment_count=original_attachment_count,
            )
            if notice_payload is not None:
                await _send_ws_payload(ws, notice_payload)
                image_notice_sent = True

        if isinstance(event, AssistantStateEvent):
            state_tracker["state"] = event.state
            if not _should_forward_state(event.state):
                logger.debug(
                    "[%s] Holding assistant state at thinking until audio is ready",
                    connection_id,
                )
                continue

            logger.debug("[%s] Assistant state -> %s", connection_id, event.state)
            await _send_ws_payload(ws, {
                "type": "assistant_state",
                "state": event.state,
            })
            continue

        if isinstance(event, AvatarExpressionEvent):
            logger.debug("[%s] Avatar expression -> %s", connection_id, event.expression)
            await _send_ws_payload(ws, {
                "type": "assistant_expression",
                "expression": event.expression,
            })
            continue

        if isinstance(event, AvatarAnimationEvent):
            logger.debug("[%s] Avatar animation -> %s", connection_id, event.animation)
            await _send_ws_payload(ws, {
                "type": "assistant_animation",
                "animation": event.animation,
            })
            continue

        if isinstance(event, AssistantThinkingEvent):
            if event.text:
                await _send_ws_payload(ws, {
                    "type": "assistant_thinking_chunk",
                    "content": event.text,
                })
            continue

        if isinstance(event, AssistantSpeechEvent):
            if not event.is_final:
                tts_chunk = thinking_filter.push(event.text)
                text_buffer += tts_chunk

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
                text_buffer += thinking_filter.flush()
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

                logger.info("[%s] Assistant turn completed", connection_id)


@app.on_event("startup")
async def startup_event():
    global tts

    app.state.server_instance_id = uuid.uuid4().hex[:8]
    app.state.orchestrator = build_orchestrator()
    app.state.memory_reflector = MemoryReflector(
        llm=app.state.orchestrator.llm,
        memory_store=app.state.orchestrator.memory_retriever.memory,
    )
    vision_config = config.raw.get("vision_watchdog", {})
    app.state.vision_watchdog = VisionWatchdog(
        model=str(vision_config.get("model", "HuggingFaceTB/SmolVLM-256M-Instruct")),
        device=str(vision_config.get("device", "auto")),
        torch_dtype=str(vision_config.get("torch_dtype", "auto")),
        attn_implementation=str(vision_config.get("attn_implementation", "auto")),
        max_new_tokens=int(vision_config.get("max_new_tokens", 8)),
        timeout_seconds=(
            float(vision_config["timeout_seconds"])
            if vision_config.get("timeout_seconds") is not None
            else None
        ),
    )
    logger.info(
        "Orchestrator initialized at startup (server_instance_id=%s)",
        app.state.server_instance_id,
    )

    tts = build_tts_engine(config.tts)
    app.state.tts_queue = asyncio.Queue(maxsize=TTS_QUEUE_MAXSIZE)
    app.state.tts_worker_task = asyncio.create_task(tts_worker(app.state.tts_queue))
    logger.info("TTS queue initialized (maxsize=%d)", TTS_QUEUE_MAXSIZE)
    app.state.stt = build_stt_engine(config.stt)


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


@app.post("/api/admin/reflect")
async def run_memory_reflection(payload: ReflectRequest):
    reflector = getattr(app.state, "memory_reflector", None)
    if reflector is None:
        orchestrator = app.state.orchestrator
        reflector = MemoryReflector(
            llm=orchestrator.llm,
            memory_store=orchestrator.memory_retriever.memory,
        )
        app.state.memory_reflector = reflector

    logger.info("Manual memory reflection requested (days_old=%d)", payload.days_old)

    result = reflector.reflect_and_prune(payload.days_old)

    if not result.get("success", True):
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Memory reflection failed",
                "error": result.get("error"),
            },
        )

    return result


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
    await ws.send_text(json.dumps(_build_session_init_payload(
        server_instance_id=server_instance_id,
        session_id=session_id,
        gesture_catalog=getattr(app.state.orchestrator, "gesture_catalog", {}),
    )))

    orchestrator = app.state.orchestrator
    watchdog = getattr(app.state, "vision_watchdog", None)
    assistant_state_tracker = {"state": AssistantState.IDLE}
    last_screen_detection = 0.0
    last_webcam_detection = 0.0
    last_screen_hash: str | None = None
    last_webcam_hash: str | None = None
    logger.debug("[%s] Reusing startup orchestrator", connection_id)

    async def handle_vision_payload(payload: dict) -> None:
        nonlocal last_screen_detection
        nonlocal last_webcam_detection
        nonlocal last_screen_hash
        nonlocal last_webcam_hash

        frame_type = payload.get("type")
        if frame_type == "screen_frame":
            key = PerceptionKey.SCREEN_SCENE
            source = "screen"
        elif frame_type == "webcam_frame":
            key = PerceptionKey.WEBCAM_SCENE
            source = "webcam"
        else:
            return

        attachment_payload = payload.get("attachment")
        if not isinstance(attachment_payload, dict):
            raise ValueError(f"{frame_type} payload is missing attachment")

        attachment = attachment_from_payload(attachment_payload)
        base64_data = getattr(attachment, "base64_data", None)
        image_hash = attachment.sha256
        if image_hash is None:
            raw_base64 = attachment_payload.get("base64_data") or attachment_payload.get("data") or ""
            image_hash = hashlib.sha256(str(raw_base64).encode("utf-8")).hexdigest()

        perception_payload = {
            **attachment.to_perception_payload(),
            "sha256": image_hash,
            "source": source,
        }
        if base64_data:
            perception_payload["base64_data"] = base64_data

        orchestrator.perception.update(
            key,
            perception_payload,
        )

        if frame_type == "screen_frame":
            if image_hash == last_screen_hash:
                return
            last_screen_hash = image_hash
            last_detection = last_screen_detection
            evaluate = watchdog.evaluate_screen if watchdog else None
        else:
            if image_hash == last_webcam_hash:
                return
            last_webcam_hash = image_hash
            last_detection = last_webcam_detection
            evaluate = watchdog.evaluate_webcam if watchdog else None

        now = time.monotonic()
        if now - last_detection < 5.0:
            return

        if assistant_state_tracker["state"] != AssistantState.IDLE:
            return

        if evaluate is None:
            logger.debug("[%s] Vision watchdog unavailable; stored %s perception only", connection_id, source)
            return

        if not base64_data:
            return

        if frame_type == "screen_frame":
            last_screen_detection = now
        else:
            last_webcam_detection = now

        should_react = await evaluate(base64_data)
        if not should_react:
            return

        if frame_type == "screen_frame":
            event_text = (
                "The local screen watchdog detected a clear visual event in the "
                "latest screenshot. Proactively help the user, briefly and concretely."
            )
        else:
            event_text = (
                "The local webcam watchdog detected that the user may need attention. "
                "Proactively check in briefly and helpfully."
            )

        logger.info("[%s] Vision watchdog triggered proactive %s turn", connection_id, source)
        await _stream_orchestrator_events(
            ws=ws,
            orchestrator=orchestrator,
            event_iterator=orchestrator.handle_proactive_event(
                session_id=session_id,
                event_text=event_text,
                attachments=[attachment],
            ),
            connection_id=connection_id,
            original_attachment_count=1,
            state_tracker=assistant_state_tracker,
        )

    try:
        while True:
            # receive() instead of receive_text() so we can handle both
            # text frames (keyboard) and binary frames (microphone audio).
            raw_message = await ws.receive()

            # Starlette surfaces disconnects as a message dict rather than
            # raising WebSocketDisconnect, so we must check before touching
            # any keys — otherwise the next receive() call crashes.
            if raw_message.get("type") == "websocket.disconnect":
                logger.info("[%s] WebSocket disconnect received", connection_id)
                break

            try:
                # ── VOICE PATH ────────────────────────────────────────────
                # Check `is not None` rather than truthiness — an empty bytes
                # value (b"") is falsy, which would wrongly fall through to
                # the text branch and crash on raw_message["text"].
                if raw_message.get("bytes") is not None:
                    stt = getattr(app.state, "stt", None)
                    if stt is None:
                        await _send_turn_error(ws, "STT is not available.")
                        continue

                    audio_bytes = raw_message["bytes"]

                    logger.info(
                        "[%s] Received audio frame (%d bytes)",
                        connection_id,
                        len(audio_bytes),
                    )

                    loop = asyncio.get_running_loop()
                    try:
                        result = await loop.run_in_executor(
                            None, stt.transcribe, audio_bytes
                        )
                    except Exception:
                        logger.exception("[%s] STT transcription failed", connection_id)
                        await _send_turn_error(ws, "Transcription failed.")
                        continue

                    if not result.text.strip():
                        logger.debug("[%s] STT returned empty transcript (silence)", connection_id)
                        await _send_ws_payload(ws, {"type": "stt_silence"})
                        continue

                    logger.info(
                        "[%s] STT transcript: %r (lang=%s)",
                        connection_id,
                        result.text,
                        result.language,
                    )

                    # Echo transcript to the UI so it can render the user bubble
                    # before the assistant starts responding.
                    await _send_ws_payload(ws, {
                        "type": "stt_transcript",
                        "text": result.text,
                        "language": result.language,
                    })

                    user_text = result.text
                    reasoning_override = None
                    attachments = []
                    input_modality = InputModality.VOICE

                # ── TEXT PATH ─────────────────────────────────────────────
                else:
                    text_payload = raw_message["text"]
                    try:
                        parsed_payload = json.loads(text_payload)
                    except json.JSONDecodeError:
                        parsed_payload = None

                    if (
                        isinstance(parsed_payload, dict)
                        and parsed_payload.get("type") in {"screen_frame", "webcam_frame"}
                    ):
                        await handle_vision_payload(parsed_payload)
                        continue

                    user_text, reasoning_override, attachments = parse_user_message(
                        text_payload
                    )
                    input_modality = InputModality.TEXT

                # ── SHARED PATH (both modalities reach here) ──────────────
                original_attachment_count = len(attachments)
                vision_attachment_count = _append_recent_vision_attachments(
                    orchestrator,
                    attachments,
                )
                if vision_attachment_count:
                    logger.debug(
                        "[%s] Added %d recent vision frame(s) to user turn context",
                        connection_id,
                        vision_attachment_count,
                    )

                logger.info(
                    "[%s] Received user input (len=%d, images=%d, reasoning_override=%r, modality=%s)",
                    connection_id,
                    len(user_text),
                    len(attachments),
                    reasoning_override,
                    input_modality.value,
                )
                logger.debug("[%s] User input text: %r", connection_id, user_text)

                await _stream_orchestrator_events(
                    ws=ws,
                    orchestrator=orchestrator,
                    event_iterator=orchestrator.handle_user_input(
                        session_id,
                        user_text,
                        think_override=reasoning_override,
                        attachments=attachments,
                        input_modality=input_modality,
                    ),
                    connection_id=connection_id,
                    original_attachment_count=original_attachment_count,
                    state_tracker=assistant_state_tracker,
                )

            except WebSocketDisconnect:
                raise
            except ValueError as exc:
                logger.warning("[%s] Rejected user message: %s", connection_id, exc)
                await _send_turn_error(
                    ws,
                    f"Couldn't process that message: {exc}",
                )
                assistant_state_tracker["state"] = AssistantState.IDLE
            except Exception:
                logger.exception("[%s] Turn failed; keeping websocket alive", connection_id)
                await _send_turn_error(
                    ws,
                    "Sorry, something went wrong while processing that message. Please try again.",
                )
                assistant_state_tracker["state"] = AssistantState.IDLE

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
