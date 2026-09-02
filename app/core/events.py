from typing import Literal, Union
from pydantic import BaseModel


class BaseEvent(BaseModel):
    """Base class for all orchestrator yield events."""
    type: str


class AssistantSpeechEvent(BaseEvent):
    type: Literal["speech"] = "speech"
    text: str
    is_final: bool = False


class UserMessageAcceptedEvent(BaseEvent):
    type: Literal["user_message_accepted"] = "user_message_accepted"
    message_id: int
    is_retry: bool = False


class AssistantTurnFailureEvent(BaseEvent):
    type: Literal["turn_failure"] = "turn_failure"
    user_message_id: int
    message: str
    attempts: int = 1


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


class AvatarOutfitEvent(BaseEvent):
    type: Literal["outfit"] = "outfit"
    outfit: str
    url: str


class AutonomyOutcomeEvent(BaseEvent):
    type: Literal["autonomy_outcome"] = "autonomy_outcome"
    summary: str
    notification: dict | None = None


# Type alias for the generator signatures!
TurnEvent = Union[
    AssistantSpeechEvent,
    UserMessageAcceptedEvent,
    AssistantTurnFailureEvent,
    AssistantThinkingEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
    AvatarAnimationEvent,
    AvatarOutfitEvent,
    AutonomyOutcomeEvent,
]
