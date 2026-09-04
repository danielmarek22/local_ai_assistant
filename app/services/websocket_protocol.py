from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class _ClientFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class UserMessageFrame(_ClientFrame):
    type: Literal["user_message"]
    text: str
    reasoning: bool | None = None
    instant_mode: bool = False
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class RetryMessageFrame(_ClientFrame):
    type: Literal["retry_message"]
    message_id: int = Field(gt=0)
    reasoning: bool | None = None
    instant_mode: bool = False


class RelayMessageFrame(_ClientFrame):
    type: Literal["relay_message"]
    sender_display_name: str
    sender_type: Literal["human", "external_agent"]
    text: str


class UserConfigFrame(_ClientFrame):
    type: Literal["user_config"]
    instant_mode: bool
    reasoning: bool | None = None


class ToolApprovalResponseFrame(_ClientFrame):
    type: Literal["tool_approval_response"]
    approval_id: str = Field(min_length=1)
    approved: bool


class VisionFrame(_ClientFrame):
    type: Literal["screen_frame", "webcam_frame", "user_attached_frame"]
    attachment: dict[str, Any]


class _ServerFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _AssistantFrame(_ServerFrame):
    turn_id: str | None = None
    origin: str | None = None


class SessionInitFrame(_ServerFrame):
    type: Literal["session_init"]
    server_instance_id: str
    session_id: str
    gesture_catalog: dict[str, str]
    outfit_catalog: dict[str, str]
    current_outfit: str | None
    session_kind: Literal["direct", "manual_group"]
    local_human_display_name: str
    local_assistant_display_name: str


class AssistantStateFrame(_AssistantFrame):
    type: Literal["assistant_state"]
    state: Literal["idle", "thinking", "dreaming", "searching", "responding"]


class AssistantExpressionFrame(_AssistantFrame):
    type: Literal["assistant_expression"]
    expression: str


class AssistantAnimationFrame(_AssistantFrame):
    type: Literal["assistant_animation"]
    animation: str


class AssistantOutfitFrame(_AssistantFrame):
    type: Literal["assistant_outfit"]
    outfit: str
    url: str


class AssistantThinkingChunkFrame(_AssistantFrame):
    type: Literal["assistant_thinking_chunk"]
    content: str


class AssistantChunkFrame(_AssistantFrame):
    type: Literal["assistant_chunk"]
    content: str


class AssistantAudioFrame(_AssistantFrame):
    type: Literal["assistant_audio"]
    url: str


class AssistantEndFrame(_AssistantFrame):
    type: Literal["assistant_end"]
    content: str


class AssistantRetryableErrorFrame(_AssistantFrame):
    type: Literal["assistant_retryable_error"]
    user_message_id: int = Field(gt=0)
    message: str
    attempts: int = Field(ge=0)


class UserMessageAcceptedFrame(_ServerFrame):
    type: Literal["user_message_accepted"]
    message_id: int = Field(gt=0)
    is_retry: bool


class UserNoticeFrame(_ServerFrame):
    type: Literal["user_notice"]
    scope: str
    tone: str
    message: str


class ToolApprovalRequestFrame(_ServerFrame):
    type: Literal["tool_approval_request"]
    approval_id: str = Field(min_length=1)
    tool: str
    title: str
    reason: str
    detail_label: str
    detail: str
    timeout_seconds: float = Field(ge=0)
    origin: str | None = None


class SttTranscriptFrame(_ServerFrame):
    type: Literal["stt_transcript"]
    text: str
    language: str | None


class SttSilenceFrame(_ServerFrame):
    type: Literal["stt_silence"]


ClientFrame = Annotated[
    UserMessageFrame
    | RetryMessageFrame
    | RelayMessageFrame
    | UserConfigFrame
    | ToolApprovalResponseFrame
    | VisionFrame,
    Field(discriminator="type"),
]
_CLIENT_FRAME_ADAPTER = TypeAdapter(ClientFrame)

ServerFrame = Annotated[
    SessionInitFrame
    | AssistantStateFrame
    | AssistantExpressionFrame
    | AssistantAnimationFrame
    | AssistantOutfitFrame
    | AssistantThinkingChunkFrame
    | AssistantChunkFrame
    | AssistantAudioFrame
    | AssistantEndFrame
    | AssistantRetryableErrorFrame
    | UserMessageAcceptedFrame
    | UserNoticeFrame
    | ToolApprovalRequestFrame
    | SttTranscriptFrame
    | SttSilenceFrame,
    Field(discriminator="type"),
]
_SERVER_FRAME_ADAPTER = TypeAdapter(ServerFrame)

CLIENT_FRAME_TYPES = frozenset(
    {
        "relay_message",
        "retry_message",
        "screen_frame",
        "tool_approval_response",
        "user_attached_frame",
        "user_config",
        "user_message",
        "webcam_frame",
    }
)
SERVER_FRAME_TYPES = frozenset(
    {
        "assistant_animation",
        "assistant_audio",
        "assistant_chunk",
        "assistant_end",
        "assistant_expression",
        "assistant_outfit",
        "assistant_retryable_error",
        "assistant_state",
        "assistant_thinking_chunk",
        "session_init",
        "stt_silence",
        "stt_transcript",
        "tool_approval_request",
        "user_message_accepted",
        "user_notice",
    }
)


def _validation_error(label: str, exc: ValidationError) -> ValueError:
    issues = []
    for error in exc.errors(include_url=False, include_input=False)[:8]:
        location = ".".join(str(part) for part in error["loc"])
        issues.append(f"{location}: {error['msg']} [{error['type']}]")
    return ValueError(f"Invalid {label}: " + "; ".join(issues))


def validate_client_frame(payload: object) -> ClientFrame:
    try:
        return _CLIENT_FRAME_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise _validation_error("client frame", exc) from exc


def validate_server_frame(payload: object) -> ServerFrame:
    try:
        return _SERVER_FRAME_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise _validation_error("server frame", exc) from exc


def encode_server_frame(payload: object) -> str:
    frame = validate_server_frame(payload)
    return frame.model_dump_json(exclude_unset=True)


def protocol_manifest() -> dict[str, object]:
    return {
        "version": 1,
        "client_frame_types": sorted(CLIENT_FRAME_TYPES),
        "server_frame_types": sorted(SERVER_FRAME_TYPES),
    }


def decode_client_frame(raw_text: str) -> ClientFrame | None:
    """Decode a typed client frame, or return None for legacy plain chat text."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or "type" not in payload:
        return None
    return validate_client_frame(payload)
