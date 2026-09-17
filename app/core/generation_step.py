"""Internal outcomes of one native generation step, before tool execution."""

from dataclasses import dataclass
from typing import Mapping, TypeAlias

from app.integrations import CapabilityId, ToolCall


@dataclass(frozen=True)
class FinalText:
    response: str


@dataclass(frozen=True)
class AcceptedToolCall:
    call: ToolCall

    @property
    def tool_name(self) -> str:
        return str(self.call.capability)

    @property
    def tool_arguments(self) -> Mapping[str, object]:
        return self.call.arguments


@dataclass(frozen=True)
class InvalidToolCall:
    # Preserve malformed values for the model's correction observation.
    tool_name: object
    tool_arguments: object
    error: str


GenerationStepResult: TypeAlias = FinalText | AcceptedToolCall | InvalidToolCall


def parse_tool_call(raw: object) -> AcceptedToolCall | InvalidToolCall:
    """Validate call structure; capability authorization belongs to the executor."""
    tool_name: object = ""
    tool_arguments: object = {}
    try:
        if not isinstance(raw, dict):
            raise ValueError("Tool call must be an object")
        function = raw.get("function", {})
        if not isinstance(function, dict):
            raise ValueError("Tool function must be an object")
        tool_name = function.get("name", "")
        tool_arguments = function.get("arguments", {})
        if not isinstance(tool_name, str):
            raise ValueError(f"Invalid capability ID: {tool_name!r}")
        capability = CapabilityId.parse(tool_name)
        if not isinstance(tool_arguments, dict):
            raise ValueError("Tool arguments must be an object")
        return AcceptedToolCall(ToolCall(capability=capability, arguments=tool_arguments))
    except (TypeError, ValueError) as exc:
        return InvalidToolCall(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            error=f"Invalid tool call {tool_name!r}: {exc}",
        )
