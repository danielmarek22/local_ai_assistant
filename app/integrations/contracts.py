from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Mapping, Protocol


_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, order=True)
class CapabilityId:
    integration: str
    action: str

    def __post_init__(self) -> None:
        for label, value in (("integration", self.integration), ("action", self.action)):
            if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
                raise ValueError(f"Invalid capability {label}: {value!r}")

    @classmethod
    def parse(cls, value: str) -> "CapabilityId":
        if not isinstance(value, str) or value.count("__") != 1:
            raise ValueError(f"Invalid capability ID: {value!r}")
        integration, action = value.split("__", 1)
        return cls(integration, action)

    def __str__(self) -> str:
        return f"{self.integration}__{self.action}"


@dataclass(frozen=True, order=True)
class EventId:
    integration: str
    event: str

    def __post_init__(self) -> None:
        for label, value in (("integration", self.integration), ("event", self.event)):
            if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
                raise ValueError(f"Invalid event {label}: {value!r}")

    @classmethod
    def parse(cls, value: str) -> "EventId":
        if not isinstance(value, str) or value.count("__") != 1:
            raise ValueError(f"Invalid event ID: {value!r}")
        integration, event = value.split("__", 1)
        return cls(integration, event)

    def __str__(self) -> str:
        return f"{self.integration}__{self.event}"


class NotificationPolicy(str, Enum):
    MODEL_DECIDES = "model_decides"
    ALWAYS_NOTIFY = "always_notify"
    NEVER_NOTIFY = "never_notify"


class ReplayPolicy(str, Enum):
    NEVER = "never"
    SAFE = "safe"


@dataclass(frozen=True)
class EventSpec:
    event: EventId
    description: str
    payload_schema: Mapping[str, object]
    allowed_capabilities: tuple[CapabilityId, ...] = ()
    notification_policy: NotificationPolicy = NotificationPolicy.MODEL_DECIDES
    replay_policy: ReplayPolicy = ReplayPolicy.NEVER
    priority: int = 100
    coalesce_window_s: float = 0.0


@dataclass(frozen=True)
class EventAttachmentRef:
    name: str
    mime_type: str
    storage_path: str
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class IntegrationEvent:
    event: EventId
    payload: Mapping[str, object]
    session_id: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None
    causation_id: str | None = None
    root_event_id: str | None = None
    deduplication_key: str | None = None
    attachments: tuple[EventAttachmentRef, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    capability: CapabilityId
    description: str
    input_schema: Mapping[str, object]


@dataclass(frozen=True)
class ToolCall:
    capability: CapabilityId
    arguments: Mapping[str, object]


class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    PENDING = "pending"


@dataclass(frozen=True)
class ToolResult:
    status: ToolResultStatus
    content: str
    operation_id: str | None = None

    @classmethod
    def success(cls, content: str) -> "ToolResult":
        return cls(ToolResultStatus.SUCCESS, content)

    @classmethod
    def error(cls, content: str) -> "ToolResult":
        return cls(ToolResultStatus.ERROR, content)

    @classmethod
    def denied(cls, content: str) -> "ToolResult":
        return cls(ToolResultStatus.DENIED, content)

    @classmethod
    def unavailable(cls, content: str) -> "ToolResult":
        return cls(ToolResultStatus.UNAVAILABLE, content)

    @classmethod
    def pending(cls, content: str, operation_id: str) -> "ToolResult":
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("Pending tool results require an operation ID")
        return cls(ToolResultStatus.PENDING, content, operation_id.strip())


@dataclass(frozen=True)
class ApprovalRequest:
    capability: CapabilityId
    title: str
    reason: str
    detail_label: str
    detail: str

    def __getitem__(self, key: str) -> str:
        return self.to_payload()[key]

    def to_payload(self) -> dict[str, str]:
        return {
            "tool": str(self.capability),
            "title": self.title,
            "reason": self.reason,
            "detail_label": self.detail_label,
            "detail": self.detail,
        }


ApprovalCallback = Callable[[ApprovalRequest], bool]


class NotificationDelivery(str, Enum):
    TEXT = "text"
    SPEECH = "speech"


@dataclass(frozen=True)
class NotificationRequest:
    message: str
    delivery: NotificationDelivery = NotificationDelivery.TEXT


NotificationCallback = Callable[[NotificationRequest], bool]


@dataclass(frozen=True)
class InvocationContext:
    session_id: str
    user_text: str
    approval_callback: ApprovalCallback | None = None
    invocation_id: str | None = None
    event_id: str | None = None
    root_event_id: str | None = None
    causation_id: str | None = None
    notification_callback: NotificationCallback | None = None


@dataclass(frozen=True)
class ContextContribution:
    source: str
    content: str


ToolHandler = Callable[[Mapping[str, object], InvocationContext], ToolResult]
AvailabilityCheck = bool | Callable[[], bool]


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler
    available: AvailabilityCheck = True

    def is_available(self) -> bool:
        return bool(self.available() if callable(self.available) else self.available)


class Integration(Protocol):
    name: str

    def registered_tools(self) -> list[RegisteredTool]: ...


EventPublisher = Callable[[IntegrationEvent], str]
