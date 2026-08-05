from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ToolResult:
    status: ToolResultStatus
    content: str

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


@dataclass(frozen=True)
class InvocationContext:
    session_id: str
    user_text: str
    approval_callback: ApprovalCallback | None = None


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
