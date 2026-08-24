from typing import Literal, Union
from pydantic import BaseModel


class BaseEvent(BaseModel):
    """Base class for all orchestrator yield events."""
    type: str


class AssistantSpeechEvent(BaseEvent):
    type: Literal["speech"] = "speech"
    text: str
    is_final: bool = False


class AssistantThinkingEvent(BaseEvent):
    type: Literal["thinking"] = "thinking"
    text: str


class AssistantStateEvent(BaseEvent):
    type: Literal["state"] = "state"
    state: str


class AvatarExpressionEvent(BaseEvent):
    type: Literal["expression"] = "expression"
    expression: str


class AvatarAnimationEvent(BaseEvent):
    type: Literal["animation"] = "animation"
    animation: str


class AutonomyOutcomeEvent(BaseEvent):
    type: Literal["autonomy_outcome"] = "autonomy_outcome"
    summary: str
    notification: dict | None = None


# Type alias for the generator signatures!
TurnEvent = Union[
    AssistantSpeechEvent,
    AssistantThinkingEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
    AvatarAnimationEvent,
    AutonomyOutcomeEvent,
]
