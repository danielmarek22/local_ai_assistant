from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class VisibilityPolicy(str, Enum):
    AGENT_CURRENT = "AGENT_CURRENT"
    SESSION_CURRENT = "SESSION_CURRENT"


class ExpiryPolicy(str, Enum):
    END_OF_SESSION = "END_OF_SESSION"
    AFTER_ONE_HOUR = "AFTER_ONE_HOUR"
    END_OF_LOCAL_DAY = "END_OF_LOCAL_DAY"
    AFTER_TWENTY_FOUR_HOURS = "AFTER_TWENTY_FOUR_HOURS"
    AFTER_SEVEN_DAYS = "AFTER_SEVEN_DAYS"
    UNTIL_EXPLICIT_DATETIME = "UNTIL_EXPLICIT_DATETIME"
    NO_AUTOMATIC_EXPIRY = "NO_AUTOMATIC_EXPIRY"


class CandidateOperation(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    INVALIDATE = "INVALIDATE"
    IGNORE = "IGNORE"


class IgnoreReason(str, Enum):
    STABLE_MEMORY = "STABLE_MEMORY"
    HYPOTHETICAL = "HYPOTHETICAL"
    NOT_CURRENT = "NOT_CURRENT"
    AMBIGUOUS = "AMBIGUOUS"
    NO_CHANGE = "NO_CHANGE"
    QUOTED_OR_EMBEDDED_CONTENT = "QUOTED_OR_EMBEDDED_CONTENT"
    META_INSTRUCTION = "META_INSTRUCTION"
    ATTRIBUTED_TO_OTHER = "ATTRIBUTED_TO_OTHER"


class _StrictCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateCandidate(_StrictCandidate):
    operation: Literal[CandidateOperation.CREATE]
    subject: str = Field(min_length=2, max_length=64)
    predicate: str = Field(min_length=2, max_length=64)
    value: Any
    visibility: VisibilityPolicy
    expiry_policy: ExpiryPolicy
    explicit_until: str | None = Field(default=None, max_length=64)
    evidence_excerpt: str | None = Field(default=None, min_length=1, max_length=500)


class UpdateCandidate(_StrictCandidate):
    operation: Literal[CandidateOperation.UPDATE]
    target_belief_id: str = Field(min_length=1, max_length=64)
    value: Any
    expiry_policy: ExpiryPolicy
    explicit_until: str | None = Field(default=None, max_length=64)
    evidence_excerpt: str | None = Field(default=None, min_length=1, max_length=500)


class InvalidateCandidate(_StrictCandidate):
    operation: Literal[CandidateOperation.INVALIDATE]
    target_belief_id: str = Field(min_length=1, max_length=64)
    evidence_excerpt: str | None = Field(default=None, min_length=1, max_length=500)


class IgnoreCandidate(_StrictCandidate):
    operation: Literal[CandidateOperation.IGNORE]
    reason: IgnoreReason
    explanation: str | None = Field(default=None, max_length=500)


BeliefCandidate = Annotated[
    Union[CreateCandidate, UpdateCandidate, InvalidateCandidate, IgnoreCandidate],
    Field(discriminator="operation"),
]


class BeliefCandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operations: list[BeliefCandidate] = Field(default_factory=list, max_length=8)


@dataclass(frozen=True)
class BeliefRecord:
    belief_id: str
    owner_agent_id: str
    visibility: VisibilityPolicy
    origin_session_id: str
    subject: str
    predicate: str
    value: Any
    confidence: float
    status: str
    expires_at: datetime | None
    source_message_id: int
    source_observed_at: datetime
    evidence_excerpt: str | None
    revision: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BeliefMutation:
    operation: CandidateOperation
    belief_id: str | None
    visibility: VisibilityPolicy | None
    origin_session_id: str
    subject: str | None
    predicate: str | None
    value: Any = None
    expires_at: datetime | None = None
    evidence_excerpt: str | None = None
