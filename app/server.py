from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, WebSocket, WebSocketDisconnect
import asyncio
from typing import Annotated, Iterator, Any
import hashlib
import json
import logging
import re
import subprocess
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
from app.core.conversation import SessionKind, relay_sender
from app.core.events import (
    AssistantSpeechEvent,
    AssistantThinkingEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
    AvatarAnimationEvent,
    AvatarOutfitEvent,
)
from app.logging import setup_logging_from_config
from app.perception.attachments import Attachment, AudioAttachment, ImageAttachment, attachment_from_payload
from app.integrations import EventAttachmentRef, EventId, IntegrationEvent
from app.perception.keys import PerceptionKey
from app.tts.factory import build_tts_engine
from app.stt.factory import build_stt_engine
from app.services.sentence_splitter import split_sentences
from app.services.memory_reflector import MemoryReflector
from app.services.vision_watchdog import VisionWatchdog
from app.services.connection_hub import SessionConnectionHub
from app.knowledge import (
    BeliefDetailDTO,
    BeliefListResponse,
    BeliefRecordStatus,
    ContextPreviewResponse,
    EffectiveBeliefsResponse,
    KnowledgeService,
    SavedMemoryListResponse,
)
from app.knowledge.models import BeliefFiltersDTO
from app.beliefs.models import EpistemicStatus, VisibilityPolicy
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
TOOL_APPROVAL_TIMEOUT_SECONDS = 300.0
BACKGROUND_VISION_FRAME_TYPES = {"screen_frame", "webcam_frame"}
VOICE_ATTACHMENT_FRAME_TYPE = "user_attached_frame"
VISION_FRAME_TYPES = BACKGROUND_VISION_FRAME_TYPES | {VOICE_ATTACHMENT_FRAME_TYPE}
VOICE_INPUT_STT = "stt"
VOICE_INPUT_NATIVE_AUDIO = "native_audio"
_PYAV_INVALID_DATA_ERRNO = "1094995529"
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


def _voice_input_path() -> str:
    path = str(config.voice_input.get("path", VOICE_INPUT_STT)).strip().lower()
    if path in {"native", "audio", "gemma4"}:
        return VOICE_INPUT_NATIVE_AUDIO
    if path in {VOICE_INPUT_STT, VOICE_INPUT_NATIVE_AUDIO}:
        return path
    logger.warning("Unknown voice_input.path=%r; falling back to %s", path, VOICE_INPUT_STT)
    return VOICE_INPUT_STT


def _native_audio_config() -> dict:
    native_audio = config.voice_input.get("native_audio", {})
    return native_audio if isinstance(native_audio, dict) else {}


def _is_invalid_stt_audio_error(exc: Exception) -> bool:
    """Return True for decoder errors caused by incomplete or non-audio blobs."""
    exc_type = type(exc)
    if exc_type.__name__ != "InvalidDataError":
        return False

    module = getattr(exc_type, "__module__", "")
    if module and not module.startswith("av"):
        return False

    message = str(exc)
    return (
        _PYAV_INVALID_DATA_ERRNO in message
        or "Invalid data found when processing input" in message
    )


def _convert_audio_to_wav(
    audio_bytes: bytes,
    *,
    sample_rate: int = 16000,
    timeout_s: float = 15.0,
) -> bytes:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Audio conversion failed: {stderr or 'ffmpeg exited unsuccessfully'}")
    return result.stdout


def _build_native_audio_attachment(audio_bytes: bytes) -> AudioAttachment:
    native_audio = _native_audio_config()
    convert_to_wav = bool(native_audio.get("convert_to_wav", True))

    if convert_to_wav:
        sample_rate = int(native_audio.get("sample_rate", 16000))
        payload = _convert_audio_to_wav(audio_bytes, sample_rate=sample_rate)
        return AudioAttachment.from_bytes(
            payload,
            name="voice.wav",
            mime_type="audio/wav",
        )

    return AudioAttachment.from_bytes(
        audio_bytes,
        name=str(native_audio.get("raw_name", "voice.webm")),
        mime_type=str(native_audio.get("raw_mime_type", "audio/webm")),
    )


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
    requested_session_exists: bool = False,
) -> str:
    if session_mode == "open" and requested_session_id:
        return requested_session_id

    if (
        session_mode == "resume"
        and requested_session_id
        and (
            known_server_instance_id == server_instance_id
            or requested_session_exists
        )
    ):
        return requested_session_id

    return uuid.uuid4().hex[:8]


def parse_user_message(raw_text: str) -> tuple[str, bool | None, bool, list[Attachment]]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text, None, False, []

    if not isinstance(payload, dict):
        return raw_text, None, False, []

    if payload.get("type") != "user_message":
        return raw_text, None, False, []

    forbidden_fields = {
        "role", "sender_id", "sender_display_name", "sender_type", "input_source",
        "target", "system", "tool", "tool_name", "tool_calls",
    }
    supplied_forbidden = sorted(forbidden_fields.intersection(payload))
    if supplied_forbidden:
        raise ValueError(
            "User message contains server-controlled fields: " + ", ".join(supplied_forbidden)
        )

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

    instant_mode = payload.get("instant_mode", False)
    if not isinstance(instant_mode, bool):
        raise ValueError("User message instant_mode flag must be boolean")

    return text, reasoning_override, instant_mode, attachments


def parse_relay_message(payload: dict, session_kind: SessionKind | str):
    if SessionKind(session_kind) != SessionKind.MANUAL_GROUP:
        raise ValueError("Relay messages are only allowed in manual_group sessions")
    allowed_fields = {"type", "sender_display_name", "sender_type", "text"}
    unexpected = sorted(set(payload) - allowed_fields)
    if unexpected:
        raise ValueError("Relay message contains unsupported fields: " + ", ".join(unexpected))
    if payload.get("type") != "relay_message":
        raise ValueError("Invalid relay message type")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Relay message text must not be empty")
    sender = relay_sender(payload.get("sender_type"), payload.get("sender_display_name"))
    return text, sender


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
    hub = getattr(app.state, "connection_hub", None)
    if hub is not None:
        await hub.send_websocket(ws, payload)
    else:
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


async def _request_tool_approval(
    ws: WebSocket,
    request: dict,
    connection_id: str,
    timeout_seconds: float = TOOL_APPROVAL_TIMEOUT_SECONDS,
) -> bool:
    approval_id = uuid.uuid4().hex
    tool_name = str(request.get("tool", "unknown"))
    title = str(request.get("title", "Approve action?"))
    reason = str(request.get("reason", "This action requires human approval."))
    detail_label = str(request.get("detail_label", "Details"))
    detail = str(request.get("detail", ""))

    logger.info("[%s] Requesting human approval for %s", connection_id, tool_name)
    await _send_ws_payload(ws, {
        "type": "tool_approval_request",
        "approval_id": approval_id,
        "tool": tool_name,
        "title": title,
        "reason": reason,
        "detail_label": detail_label,
        "detail": detail,
        "timeout_seconds": timeout_seconds,
    })

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("[%s] Tool approval timed out for %s", connection_id, tool_name)
            return False

        raw_message = await asyncio.wait_for(ws.receive(), timeout=remaining)
        if raw_message.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect()

        text_payload = raw_message.get("text")
        if text_payload is None:
            continue

        try:
            payload = json.loads(text_payload)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        if (
            payload.get("type") != "tool_approval_response"
            or payload.get("approval_id") != approval_id
        ):
            logger.debug("[%s] Ignoring websocket message while awaiting tool approval", connection_id)
            continue

        approved = bool(payload.get("approved"))
        logger.info(
            "[%s] Human %s capability %s",
            connection_id,
            "approved" if approved else "denied",
            tool_name,
        )
        return approved


def _build_session_init_payload(
    server_instance_id: str,
    session_id: str,
    gesture_catalog: dict[str, str] | None = None,
    outfit_catalog: dict[str, str] | None = None,
    current_outfit: str | None = None,
    session_kind: SessionKind | str = SessionKind.DIRECT,
    local_human_display_name: str = "You",
    local_assistant_display_name: str = "Astra",
) -> dict:
    return {
        "type": "session_init",
        "server_instance_id": server_instance_id,
        "session_id": session_id,
        "gesture_catalog": dict(gesture_catalog or {}),
        "outfit_catalog": dict(outfit_catalog or {}),
        "current_outfit": current_outfit,
        "session_kind": SessionKind(session_kind).value,
        "local_human_display_name": local_human_display_name,
        "local_assistant_display_name": local_assistant_display_name,
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


def _dedupe_attachments_by_hash(attachments: list[Attachment]) -> list[Attachment]:
    deduped: list[Attachment] = []
    seen_hashes: set[str] = set()

    for attachment in attachments:
        attachment_hash = getattr(attachment, "sha256", None)
        if attachment_hash:
            if attachment_hash in seen_hashes:
                continue
            seen_hashes.add(attachment_hash)
        deduped.append(attachment)

    return deduped


def _persist_event_attachment(event_id: str, attachment: ImageAttachment) -> EventAttachmentRef:
    event_dir = Path("static/uploads/events") / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    suffix = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }.get(attachment.mime_type, ".bin")
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(attachment.name).stem) or "attachment"
    path = event_dir / f"{safe_stem}{suffix}"
    payload = attachment.as_bytes()
    path.write_bytes(payload)
    return EventAttachmentRef(
        name=attachment.name,
        mime_type=attachment.mime_type,
        storage_path=str(path),
        sha256=attachment.sha256 or hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


async def _stream_orchestrator_events(
    ws: WebSocket,
    orchestrator,
    event_iterator: Iterator[Any],
    connection_id: str,
    original_attachment_count: int,
    state_tracker: dict[str, str],
) -> None:
    hub = getattr(app.state, "connection_hub", None)
    turn_id = uuid.uuid4().hex
    if hub is not None:
        hub.set_turn(connection_id, turn_id, "user")
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

        if isinstance(event, AvatarOutfitEvent):
            logger.info("[%s] Avatar outfit -> %s", connection_id, event.outfit)
            await _send_ws_payload(ws, {
                "type": "assistant_outfit",
                "outfit": event.outfit,
                "url": event.url,
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

    if hub is not None:
        hub.set_turn(connection_id, None, None)


async def _autonomy_output_sink(session_id: str, event, turn_id: str) -> None:
    if not isinstance(event, AssistantStateEvent):
        return
    hub = getattr(app.state, "connection_hub", None)
    if hub is not None:
        await hub.broadcast(session_id, {
            "type": "assistant_state",
            "state": event.state,
            "turn_id": turn_id,
            "origin": "integration_event",
        })


async def _autonomy_notification_sink(
    session_id: str,
    notification: dict[str, object],
    turn_id: str,
) -> None:
    hub = getattr(app.state, "connection_hub", None)
    if hub is None or not hub.has_session(session_id):
        return
    message = str(notification.get("message", "")).strip()
    if not message:
        return
    await hub.broadcast(session_id, {
        "type": "assistant_chunk",
        "content": message,
        "turn_id": turn_id,
        "origin": "integration_event",
    })
    if notification.get("delivery") == "speech" and tts is not None:
        audio_id = uuid.uuid4().hex
        audio_path = AUDIO_DIR / f"{audio_id}.wav"
        try:
            await synthesize_async(message, audio_path, session_id)
        except Exception:
            logger.exception("[%s] Autonomous notification TTS failed", session_id)
        else:
            await hub.broadcast(session_id, {
                "type": "assistant_audio",
                "url": f"/static/audio/{audio_id}.wav",
                "turn_id": turn_id,
                "origin": "integration_event",
            })
    await hub.broadcast(session_id, {
        "type": "assistant_end",
        "content": message,
        "turn_id": turn_id,
        "origin": "integration_event",
    })


async def _autonomy_approval_provider(session_id: str, request: dict[str, object]) -> bool:
    hub = getattr(app.state, "connection_hub", None)
    if hub is None:
        return False
    return await hub.request_approval(
        session_id,
        request,
        timeout_seconds=float(config.autonomy.get("approval_timeout_s", 300)),
    )


@app.on_event("startup")
async def startup_event():
    global tts

    app.state.server_instance_id = uuid.uuid4().hex[:8]
    app.state.connection_hub = SessionConnectionHub()
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
    app.state.voice_input_path = _voice_input_path()
    if app.state.voice_input_path == VOICE_INPUT_STT:
        app.state.stt = build_stt_engine(config.stt)
    else:
        app.state.stt = None
    logger.info("Voice input path configured as %s", app.state.voice_input_path)
    autonomy_runtime = getattr(app.state.orchestrator, "autonomy_runtime", None)
    if autonomy_runtime is not None:
        autonomy_runtime.output_sink = _autonomy_output_sink
        autonomy_runtime.notification_sink = _autonomy_notification_sink
        autonomy_runtime.approval_provider = _autonomy_approval_provider
        await autonomy_runtime.start()
        app.state.autonomy_runtime = autonomy_runtime
        logger.info("Autonomy runtime started")


@app.on_event("shutdown")
async def shutdown_event():
    queue = getattr(app.state, "tts_queue", None)
    worker_task = getattr(app.state, "tts_worker_task", None)
    orchestrator = getattr(app.state, "orchestrator", None)
    autonomy_runtime = getattr(app.state, "autonomy_runtime", None)

    if autonomy_runtime is not None:
        await autonomy_runtime.close()
    else:
        close_orchestrator = getattr(orchestrator, "close", None)
        if callable(close_orchestrator):
            close_orchestrator()

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
            "kind": row["kind"],
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

    if not rows and not history_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    summary = app.state.orchestrator.summary_store.get(session_id)
    summary_text = summary[0] if summary else None
    return {
        "session_id": session_id,
        "kind": history_store.get_session_kind(session_id).value,
        "summary": summary_text,
        "messages": [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "attachments": [attachment.to_api_payload() for attachment in row.get("attachments", [])],
                "sender_id": row["sender_id"],
                "sender_display_name": row["sender_display_name"],
                "sender_type": row["sender_type"],
                "input_source": row["input_source"],
            }
            for row in rows
        ],
    }


def _knowledge_service(
    *,
    require_beliefs: bool = True,
    require_memories: bool = False,
) -> KnowledgeService:
    orchestrator = app.state.orchestrator
    repository = getattr(orchestrator, "belief_repository", None)
    provider = getattr(orchestrator, "belief_context_provider", None)
    memory_retriever = getattr(orchestrator, "memory_retriever", None)
    memory_store = getattr(memory_retriever, "memory", None)
    if require_beliefs and (repository is None or provider is None):
        raise HTTPException(status_code=503, detail="Knowledge subsystem is unavailable")
    if require_memories and memory_store is None:
        raise HTTPException(status_code=503, detail="Saved memory storage is unavailable")
    return KnowledgeService(
        owner_agent_id=orchestrator.agent_id,
        repository=repository,
        context_provider=provider,
        history_store=orchestrator.history,
        memory_store=memory_store,
    )


SessionIdQuery = Annotated[
    str,
    Query(
        min_length=1,
        max_length=128,
        pattern=r"^[^\x00-\x1f\x7f]+$",
    ),
]


def _require_known_session(service: KnowledgeService, session_id: str) -> None:
    if not service.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/knowledge/memories", response_model=SavedMemoryListResponse)
async def list_saved_memories():
    return _knowledge_service(
        require_beliefs=False,
        require_memories=True,
    ).list_saved_memories()


@app.get(
    "/api/knowledge/beliefs/effective",
    response_model=EffectiveBeliefsResponse,
)
async def get_effective_beliefs(session_id: SessionIdQuery):
    service = _knowledge_service()
    _require_known_session(service, session_id)
    return service.effective_beliefs(session_id)


@app.get("/api/knowledge/beliefs", response_model=BeliefListResponse)
async def list_beliefs_for_inspection(
    subject_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    source_sender_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    predicate: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    epistemic_status: EpistemicStatus | None = None,
    visibility: VisibilityPolicy | None = None,
    record_status: BeliefRecordStatus | None = None,
    scope_session_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    source_session_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=100000)] = 0,
):
    filters = BeliefFiltersDTO(
        subject_id=subject_id,
        source_sender_id=source_sender_id,
        predicate=predicate,
        epistemic_status=epistemic_status,
        visibility=visibility,
        record_status=record_status,
        scope_session_id=scope_session_id,
        source_session_id=source_session_id,
    )
    return _knowledge_service().list_beliefs(
        filters=filters,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/api/knowledge/beliefs/{belief_id}",
    response_model=BeliefDetailDTO,
)
async def get_belief_for_inspection(
    belief_id: Annotated[
        str,
        ApiPath(min_length=1, max_length=64, pattern=r"^[^\x00-\x1f\x7f]+$"),
    ],
):
    detail = _knowledge_service().get_belief_detail(belief_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Belief not found")
    return detail


@app.get(
    "/api/knowledge/belief-context",
    response_model=ContextPreviewResponse,
)
async def get_belief_context_preview(session_id: SessionIdQuery):
    service = _knowledge_service()
    _require_known_session(service, session_id)
    return service.context_preview(session_id)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    orchestrator = app.state.orchestrator
    deleted_count = orchestrator.history.delete_session(session_id)
    orchestrator.summary_store.delete(session_id)
    belief_repository = getattr(orchestrator, "belief_repository", None)
    if belief_repository is not None:
        belief_repository.delete_session(orchestrator.agent_id, session_id)

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


@app.get("/api/autonomy")
async def get_autonomy_status():
    runtime = getattr(app.state, "autonomy_runtime", None)
    if runtime is None:
        return {"enabled": False, "paused": True, "queued": 0, "running": False}
    return runtime.status()


class AutonomyStateRequest(BaseModel):
    paused: bool


@app.put("/api/autonomy")
async def set_autonomy_status(payload: AutonomyStateRequest):
    runtime = getattr(app.state, "autonomy_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Autonomy runtime is unavailable")
    await runtime.set_paused(payload.paused)
    return runtime.status()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    connection_id = uuid.uuid4().hex[:8]
    start_ts = time.perf_counter()
    server_instance_id = app.state.server_instance_id
    session_mode = ws.query_params.get("session_mode", "new")
    requested_session_id = ws.query_params.get("session_id")
    known_server_instance_id = ws.query_params.get("server_instance_id")
    requested_session_kind_value = ws.query_params.get("session_kind", SessionKind.DIRECT.value)
    try:
        requested_session_kind = SessionKind(requested_session_kind_value)
    except ValueError:
        requested_session_kind = SessionKind.DIRECT

    orchestrator = app.state.orchestrator
    history_store = orchestrator.history
    session_id = resolve_session_id(
        session_mode=session_mode,
        requested_session_id=requested_session_id,
        known_server_instance_id=known_server_instance_id,
        server_instance_id=server_instance_id,
        requested_session_exists=bool(
            requested_session_id
            and history_store.session_exists(requested_session_id)
        ),
    )

    if requested_session_id and session_id == requested_session_id:
        session_kind = history_store.get_session_kind(session_id)
    else:
        session_kind = history_store.ensure_session(session_id, requested_session_kind)

    await ws.accept()
    hub = app.state.connection_hub
    hub.register(session_id, connection_id, ws)
    logger.info(
        "[%s] WebSocket connected (conversation_session=%s, mode=%s)",
        connection_id,
        session_id,
        session_mode,
    )
    await _send_ws_payload(ws, _build_session_init_payload(
        server_instance_id=server_instance_id,
        session_id=session_id,
        gesture_catalog=getattr(app.state.orchestrator, "gesture_catalog", {}),
        outfit_catalog=getattr(
            getattr(orchestrator, "avatar_wardrobe", None), "catalog", {}
        ),
        current_outfit=getattr(
            getattr(orchestrator, "avatar_wardrobe", None), "current_outfit", None
        ),
        session_kind=session_kind,
        local_human_display_name=getattr(orchestrator, "local_human_name", "You"),
        local_assistant_display_name=getattr(orchestrator, "local_assistant_name", "Astra"),
    ))

    watchdog = getattr(app.state, "vision_watchdog", None)
    event_loop = asyncio.get_running_loop()
    assistant_state_tracker = {"state": AssistantState.IDLE}
    last_screen_detection = 0.0
    last_webcam_detection = 0.0
    last_screen_hash: str | None = None
    last_webcam_hash: str | None = None
    pending_voice_attachments: list[Attachment] = []
    connection_instant_mode = False
    connection_reasoning_override: bool | None = None
    logger.debug("[%s] Reusing startup orchestrator", connection_id)

    def request_tool_approval(request: dict) -> bool:
        future = asyncio.run_coroutine_threadsafe(
            _request_tool_approval(
                ws=ws,
                request=request,
                connection_id=connection_id,
            ),
            event_loop,
        )
        try:
            return bool(future.result(timeout=TOOL_APPROVAL_TIMEOUT_SECONDS + 5.0))
        except Exception:
            logger.exception("[%s] Tool approval failed; denying request", connection_id)
            return False

    async def handle_vision_payload(payload: dict) -> Attachment | None:
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
        elif frame_type == VOICE_ATTACHMENT_FRAME_TYPE:
            key = None
            source = "voice_attachment"
        else:
            return None

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

        if key is not None:
            orchestrator.perception.update(
                key,
                perception_payload,
            )

        if frame_type == VOICE_ATTACHMENT_FRAME_TYPE:
            return attachment

        if frame_type == "screen_frame":
            if image_hash == last_screen_hash:
                return None
            last_screen_hash = image_hash
            last_detection = last_screen_detection
            evaluate = watchdog.evaluate_screen if watchdog else None
        else:
            if image_hash == last_webcam_hash:
                return None
            last_webcam_hash = image_hash
            last_detection = last_webcam_detection
            evaluate = watchdog.evaluate_webcam if watchdog else None

        now = time.monotonic()
        if now - last_detection < 5.0:
            return None

        if evaluate is None:
            logger.debug("[%s] Vision watchdog unavailable; stored %s perception only", connection_id, source)
            return None

        if not base64_data:
            return None

        if frame_type == "screen_frame":
            last_screen_detection = now
        else:
            last_webcam_detection = now

        should_react = await evaluate(base64_data)
        if not should_react:
            return None

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

        runtime = getattr(orchestrator, "autonomy_runtime", None)
        if runtime is None:
            logger.warning("[%s] Vision event ignored because autonomy is unavailable", connection_id)
            return None
        event_id = str(uuid.uuid4())
        event_attachment = _persist_event_attachment(event_id, attachment)
        await runtime.publish(IntegrationEvent(
            event=EventId("vision", "attention_detected"),
            event_id=event_id,
            session_id=session_id,
            payload={
                "source": source,
                "description": event_text,
                "sha256": image_hash,
            },
            deduplication_key=f"{source}:{image_hash}",
            attachments=(event_attachment,),
        ))
        logger.info("[%s] Vision watchdog published autonomous %s event", connection_id, source)
        return None

    try:
        while True:
            # receive() instead of receive_text() so we can handle both
            # text frames (keyboard) and binary frames (microphone audio).
            raw_message = await ws.receive()
            hub.touch(connection_id)

            # Starlette surfaces disconnects as a message dict rather than
            # raising WebSocketDisconnect, so we must check before touching
            # any keys — otherwise the next receive() call crashes.
            if raw_message.get("type") == "websocket.disconnect":
                logger.info("[%s] WebSocket disconnect received", connection_id)
                break

            try:
                turn_sender = None
                # ── VOICE PATH ────────────────────────────────────────────
                # Check `is not None` rather than truthiness — an empty bytes
                # value (b"") is falsy, which would wrongly fall through to
                # the text branch and crash on raw_message["text"].
                if raw_message.get("bytes") is not None:
                    audio_bytes = raw_message["bytes"]
                    voice_input_path = getattr(app.state, "voice_input_path", _voice_input_path())

                    if voice_input_path == VOICE_INPUT_NATIVE_AUDIO:
                        logger.info(
                            "[%s] Received native audio frame (%d bytes)",
                            connection_id,
                            len(audio_bytes),
                        )
                        loop = asyncio.get_running_loop()
                        try:
                            audio_attachment = await loop.run_in_executor(
                                None,
                                _build_native_audio_attachment,
                                audio_bytes,
                            )
                        except Exception:
                            logger.exception("[%s] Native audio preparation failed", connection_id)
                            pending_voice_attachments = []
                            await _send_turn_error(ws, "Audio preparation failed.")
                            continue

                        native_audio = _native_audio_config()
                        display_text = str(native_audio.get("display_text", "Voice message"))
                        user_text = str(
                            native_audio.get(
                                "prompt_text",
                                "Please answer the user's spoken audio.",
                            )
                        )

                        await _send_ws_payload(ws, {
                            "type": "stt_transcript",
                            "text": display_text,
                            "language": None,
                        })

                        reasoning_override = connection_reasoning_override
                        instant_mode = connection_instant_mode
                        attachments = [audio_attachment]
                        input_modality = InputModality.VOICE

                    else:
                        stt = getattr(app.state, "stt", None)
                        if stt is None:
                            pending_voice_attachments = []
                            await _send_turn_error(ws, "STT is not available.")
                            continue

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
                        except Exception as exc:
                            if _is_invalid_stt_audio_error(exc):
                                logger.debug(
                                    "[%s] Ignoring undecodable STT audio frame (%d bytes): %s",
                                    connection_id,
                                    len(audio_bytes),
                                    exc,
                                )
                                pending_voice_attachments = []
                                await _send_ws_payload(ws, {"type": "stt_silence"})
                                continue

                            logger.exception("[%s] STT transcription failed", connection_id)
                            pending_voice_attachments = []
                            await _send_turn_error(ws, "Transcription failed.")
                            continue

                        if not result.text.strip():
                            logger.debug("[%s] STT returned empty transcript (silence)", connection_id)
                            pending_voice_attachments = []
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
                        reasoning_override = connection_reasoning_override
                        instant_mode = connection_instant_mode
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
                        and parsed_payload.get("type") == "tool_approval_response"
                        and hub.resolve_approval(connection_id, parsed_payload)
                    ):
                        continue

                    if (
                        isinstance(parsed_payload, dict)
                        and parsed_payload.get("type") in VISION_FRAME_TYPES
                    ):
                        attachment = await handle_vision_payload(parsed_payload)
                        if attachment is not None:
                            pending_voice_attachments.append(attachment)
                            pending_voice_attachments = _dedupe_attachments_by_hash(
                                pending_voice_attachments
                            )[-4:]
                        continue

                    if (
                        isinstance(parsed_payload, dict)
                        and parsed_payload.get("type") == "user_config"
                    ):
                        instant_value = parsed_payload.get("instant_mode")
                        if not isinstance(instant_value, bool):
                            raise ValueError("User config instant_mode flag must be boolean")
                        connection_instant_mode = instant_value
                        if "reasoning" in parsed_payload:
                            reasoning_value = parsed_payload.get("reasoning")
                            if reasoning_value is None:
                                connection_reasoning_override = None
                            elif isinstance(reasoning_value, bool):
                                connection_reasoning_override = reasoning_value
                            else:
                                raise ValueError("User config reasoning flag must be boolean or null")
                        continue

                    if (
                        isinstance(parsed_payload, dict)
                        and parsed_payload.get("type") == "relay_message"
                    ):
                        user_text, turn_sender = parse_relay_message(parsed_payload, session_kind)
                        reasoning_override = connection_reasoning_override
                        instant_mode = connection_instant_mode
                        attachments = []
                    else:
                        user_text, reasoning_override, instant_mode, attachments = parse_user_message(
                            text_payload
                        )
                    pending_voice_attachments = []
                    input_modality = InputModality.TEXT

                # ── SHARED PATH (both modalities reach here) ──────────────
                if input_modality == InputModality.VOICE and pending_voice_attachments:
                    attachments = _dedupe_attachments_by_hash(
                        [
                            *pending_voice_attachments,
                            *attachments,
                        ]
                    )
                    pending_voice_attachments = []

                original_attachment_count = len(attachments)
                vision_attachment_count = 0
                if turn_sender is None:
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

                runtime = getattr(orchestrator, "autonomy_runtime", None)
                turn_context = runtime.coordinator.user_turn(session_id) if runtime else None
                if turn_context is None:
                    await _stream_orchestrator_events(
                        ws, orchestrator,
                        orchestrator.handle_user_input(
                            session_id, user_text, think_override=reasoning_override,
                            instant_mode=instant_mode, attachments=attachments,
                            input_modality=input_modality,
                            tool_approval_callback=request_tool_approval,
                            sender=turn_sender,
                            session_kind=session_kind,
                        ),
                        connection_id, original_attachment_count, assistant_state_tracker,
                    )
                else:
                    async with turn_context:
                        await _stream_orchestrator_events(
                            ws, orchestrator,
                            orchestrator.handle_user_input(
                                session_id, user_text, think_override=reasoning_override,
                                instant_mode=instant_mode, attachments=attachments,
                                input_modality=input_modality,
                                tool_approval_callback=request_tool_approval,
                                sender=turn_sender,
                                session_kind=session_kind,
                            ),
                            connection_id, original_attachment_count, assistant_state_tracker,
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
        hub.unregister(connection_id)
        logger.debug("[%s] WebSocket cleanup complete", connection_id)

@app.get("/")
async def get_index():
    logger.debug("Serving index.html")
    return FileResponse("static/index.html")
