from __future__ import annotations

import json
from datetime import datetime, timezone

from app.knowledge.models import (
    BeliefDetailDTO,
    BeliefFiltersDTO,
    BeliefListResponse,
    BeliefRecordStatus,
    BeliefSummaryDTO,
    ContextPreviewResponse,
    EffectiveBeliefsResponse,
    SavedMemoryDTO,
    SavedMemoryListResponse,
    SourceDTO,
    SubjectDTO,
)
from app.services.context_builder import render_belief_context_section


_PARSE_ERROR = "Stored value is not valid JSON."


class KnowledgeService:
    """Read-only facade over persisted knowledge and the production context provider."""

    def __init__(
        self,
        *,
        owner_agent_id,
        repository,
        context_provider,
        history_store,
        memory_store=None,
    ):
        self.owner_agent_id = owner_agent_id
        self.repository = repository
        self.context_provider = context_provider
        self.history_store = history_store
        self.memory_store = memory_store

    def session_exists(self, session_id: str) -> bool:
        return bool(self.history_store.session_exists(session_id))

    def list_beliefs(
        self,
        *,
        filters: BeliefFiltersDTO,
        limit: int,
        offset: int,
    ) -> BeliefListResponse:
        now = datetime.now(timezone.utc)
        filter_values = filters.model_dump(mode="json")
        rows, total = self.repository.list_for_inspection(
            self.owner_agent_id,
            **filter_values,
            limit=limit,
            offset=offset,
            now=now,
        )
        return BeliefListResponse(
            records=[self._summary_from_row(row, now) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
            applied_filters=filters,
        )

    def get_belief_detail(self, belief_id: str) -> BeliefDetailDTO | None:
        row = self.repository.get_for_inspection(self.owner_agent_id, belief_id)
        if row is None:
            return None
        now = datetime.now(timezone.utc)
        summary = self._summary_from_row(row, now)
        return BeliefDetailDTO(
            **summary.model_dump(),
            owner_agent_id=row["owner_agent_id"],
            evidence_excerpt=row["evidence_excerpt"],
            confidence=float(row["confidence"]),
            stored_status=row["status"],
            is_expired=self._is_expired(row, now),
            created_at=datetime.fromisoformat(row["created_at"]),
            observed_at=datetime.fromisoformat(row["source_observed_at"]),
        )

    def effective_beliefs(self, session_id: str) -> EffectiveBeliefsResponse:
        preview = self.context_provider.preview_for_turn(session_id)
        records = [self._summary_from_record(record) for record in preview.beliefs]
        return EffectiveBeliefsResponse(session_id=session_id, records=records)

    def context_preview(self, session_id: str) -> ContextPreviewResponse:
        preview = self.context_provider.preview_for_turn(session_id)
        text = render_belief_context_section(preview.formatted_body)
        return ContextPreviewResponse(
            session_id=session_id,
            state="formatted" if text else "empty",
            text=text,
        )

    def list_saved_memories(self) -> SavedMemoryListResponse:
        rows = self.memory_store.list_for_inspection()
        records = [SavedMemoryDTO(**row) for row in rows]
        return SavedMemoryListResponse(records=records, total=len(records))

    @staticmethod
    def _parse_value(value_json: str):
        try:
            return json.loads(value_json), None
        except (json.JSONDecodeError, TypeError):
            return None, _PARSE_ERROR

    @staticmethod
    def _is_expired(row: dict, now: datetime) -> bool:
        if row["status"] != "active" or not row["expires_at"]:
            return False
        return datetime.fromisoformat(row["expires_at"]) <= now

    @classmethod
    def _record_status(cls, row: dict, now: datetime) -> BeliefRecordStatus:
        if row["status"] == "invalidated":
            return BeliefRecordStatus.INVALIDATED
        if cls._is_expired(row, now):
            return BeliefRecordStatus.EXPIRED
        return BeliefRecordStatus.ACTIVE

    @classmethod
    def _summary_from_row(cls, row: dict, now: datetime) -> BeliefSummaryDTO:
        value, parse_error = cls._parse_value(row["value_json"])
        return BeliefSummaryDTO(
            belief_id=row["belief_id"],
            record_status=cls._record_status(row, now),
            subject=SubjectDTO(
                id=row["subject_id"],
                kind=row["subject_kind"],
                display_name_at_evidence_time=row["subject_display_name"],
            ),
            predicate=row["predicate"],
            value=value,
            value_json=row["value_json"],
            value_parse_error=parse_error,
            epistemic_status=row["epistemic_status"],
            source=SourceDTO(
                sender_id=row["source_sender_id"],
                sender_type=row["source_sender_type"],
                input_source=row["source_input_source"],
                display_name_at_evidence_time=row["source_sender_display_name"],
            ),
            visibility=row["visibility"],
            scope_session_id=row["scope_session_id"] or None,
            source_session_id=row["source_session_id"],
            expires_at=(
                datetime.fromisoformat(row["expires_at"])
                if row["expires_at"] else None
            ),
            revision=int(row["revision"]),
            source_message_id=int(row["source_message_id"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _summary_from_record(record) -> BeliefSummaryDTO:
        value_json = json.dumps(
            record.value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return BeliefSummaryDTO(
            belief_id=record.belief_id,
            record_status=BeliefRecordStatus.ACTIVE,
            subject=SubjectDTO(
                id=record.subject_id,
                kind=record.subject_kind,
                display_name_at_evidence_time=record.subject_display_name,
            ),
            predicate=record.predicate,
            value=record.value,
            value_json=value_json,
            epistemic_status=record.epistemic_status,
            source=SourceDTO(
                sender_id=record.source_sender_id,
                sender_type=record.source_sender_type,
                input_source=record.source_input_source,
                display_name_at_evidence_time=record.source_sender_display_name,
            ),
            visibility=record.visibility,
            scope_session_id=record.scope_session_id or None,
            source_session_id=record.source_session_id,
            expires_at=record.expires_at,
            revision=record.revision,
            source_message_id=record.source_message_id,
            updated_at=record.updated_at,
        )
