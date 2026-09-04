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


def validate_client_frame(payload: object) -> ClientFrame:
    try:
        return _CLIENT_FRAME_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        issues = []
        for error in exc.errors(include_url=False, include_input=False)[:8]:
            location = ".".join(str(part) for part in error["loc"])
            issues.append(f"{location}: {error['msg']} [{error['type']}]")
        raise ValueError("Invalid client frame: " + "; ".join(issues)) from exc


def decode_client_frame(raw_text: str) -> ClientFrame | None:
    """Decode a typed client frame, or return None for legacy plain chat text."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or "type" not in payload:
        return None
    return validate_client_frame(payload)
