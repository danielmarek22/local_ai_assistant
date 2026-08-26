from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.beliefs.models import EpistemicStatus, SubjectKind, VisibilityPolicy


class BeliefRecordStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


class SubjectDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: SubjectKind
    display_name_at_evidence_time: str


class SourceDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sender_id: str
    sender_type: str
    input_source: str
    display_name_at_evidence_time: str


class BeliefSummaryDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    belief_id: str
    record_status: BeliefRecordStatus
    subject: SubjectDTO
    predicate: str
    value: Any
    value_json: str
    value_parse_error: str | None = None
    epistemic_status: EpistemicStatus
    source: SourceDTO
    visibility: VisibilityPolicy
    scope_session_id: str | None
    source_session_id: str
    expires_at: datetime | None
    revision: int
    source_message_id: int
    updated_at: datetime


class BeliefDetailDTO(BeliefSummaryDTO):
    owner_agent_id: str
    evidence_excerpt: str | None
    confidence: float
    stored_status: str
    is_expired: bool
    created_at: datetime
    observed_at: datetime


class BeliefFiltersDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_id: str | None = None
    source_sender_id: str | None = None
    predicate: str | None = None
    epistemic_status: EpistemicStatus | None = None
    visibility: VisibilityPolicy | None = None
    record_status: BeliefRecordStatus | None = None
    scope_session_id: str | None = None
    source_session_id: str | None = None


class BeliefListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[BeliefSummaryDTO]
    total: int
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    applied_filters: BeliefFiltersDTO


class EffectiveBeliefsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    records: list[BeliefSummaryDTO]


class ContextPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    state: Literal["empty", "formatted"]
    text: str


class SavedMemoryDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: str | None
    content: str
    importance: int | None
    created_at: datetime | None
    last_accessed_at: datetime | None


class SavedMemoryListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[SavedMemoryDTO]
    total: int = Field(ge=0)
