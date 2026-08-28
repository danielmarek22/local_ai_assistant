from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.beliefs.models import (
    AssertionCandidate,
    BeliefCandidateBatch,
    ExpiryPolicy,
    InvalidateCandidate,
    VisibilityPolicy,
)
from app.beliefs.preparation import (
    PreparedBeliefTurn,
    is_conversational_belief_turn_eligible,
)
from app.beliefs.subjects import resolve_subject_reference
from app.beliefs.version import REACT_TOOL_BELIEF_VERSION
from app.integrations.contracts import (
    CapabilityId,
    InvocationContext,
    RegisteredTool,
    ToolResult,
    ToolSpec,
)


BELIEF_TOOL_DESCRIPTION = (
    "Update Astra’s current, revisable understanding of what is true, using "
    "explicit evidence from the authoritative current participant message. "
    "Call at most once for a message and include the complete mutation batch. "
    "Do not use for Astra’s opinions, persona facts, generated conclusions, "
    "tool results, integration state, or durable narrative memory."
)


class _StrictToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BeliefToolAssertion(_StrictToolInput):
    subject_reference: str = Field(min_length=1, max_length=128)
    predicate: str = Field(min_length=2, max_length=64)
    value: Any
    visibility: VisibilityPolicy
    expiry_policy: ExpiryPolicy
    explicit_until: str | None = Field(default=None, max_length=64)
    evidence_excerpt: str = Field(min_length=1, max_length=500)


class BeliefToolInvalidation(_StrictToolInput):
    target_belief_id: str = Field(min_length=1, max_length=64)
    evidence_excerpt: str = Field(min_length=1, max_length=500)


class BeliefToolBatch(_StrictToolInput):
    assertions: list[BeliefToolAssertion]
    invalidations: list[BeliefToolInvalidation]

    @model_validator(mode="after")
    def require_operation(self):
        if not self.assertions and not self.invalidations:
            raise ValueError("at least one assertion or invalidation is required")
        return self


class BeliefIntegration:
    name = "beliefs"

    def __init__(self, *, update_service, max_candidates: int = 4):
        self.update_service = update_service
        self.max_candidates = max(1, min(int(max_candidates), 8))

    def registered_tools(self) -> list[RegisteredTool]:
        return [RegisteredTool(
            spec=ToolSpec(
                capability=CapabilityId(self.name, "update"),
                description=BELIEF_TOOL_DESCRIPTION,
                input_schema=self._schema(),
            ),
            handler=self._update,
            exposure_check=self._is_exposed,
        )]

    @staticmethod
    def _is_exposed(context: InvocationContext | None) -> bool:
        return bool(
            context is not None
            and is_conversational_belief_turn_eligible(context.authoritative_turn)
            and isinstance(context.prepared_belief_turn, PreparedBeliefTurn)
            and context.prepared_belief_turn.authoritative_turn == context.authoritative_turn
            and (
                context.prepared_belief_turn.grounded_subject_references()
                or context.prepared_belief_turn.permitted_invalidation_ids
            )
        )

    def _update(
        self,
        arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        turn = context.authoritative_turn
        prepared = context.prepared_belief_turn
        if not is_conversational_belief_turn_eligible(turn):
            return self._rejection(
                "authority_eligibility",
                "beliefs__update requires an eligible authoritative participant turn.",
                repository_accessed=False,
            )
        if not isinstance(prepared, PreparedBeliefTurn) or prepared.authoritative_turn != turn:
            return self._rejection(
                "frozen_catalog_validation",
                "beliefs__update has no matching frozen authoritative turn scope.",
                repository_accessed=False,
            )
        if self.update_service.repository.has_application(
            turn.owner_agent_id, turn.user_message_id, REACT_TOOL_BELIEF_VERSION
        ):
            return ToolResult.success(
                "Belief changes for this participant message were already processed. "
                "Do not call beliefs__update again for this message."
            )
        try:
            wire = BeliefToolBatch.model_validate(dict(arguments))
        except ValidationError as exc:
            return self._rejection(
                "dto_decoding",
                self._validation_summary(exc),
                repository_accessed=True,
            )
        total = len(wire.assertions) + len(wire.invalidations)
        if total > self.max_candidates:
            return self._rejection(
                "candidate_normalization",
                f"Belief update rejected: combined operation count exceeds {self.max_candidates}.",
                repository_accessed=True,
            )

        permitted_ids = prepared.permitted_invalidation_ids
        operations = []
        try:
            for index, item in enumerate(wire.assertions):
                if (
                    item.expiry_policy == ExpiryPolicy.END_OF_SESSION
                    and item.visibility != VisibilityPolicy.SESSION_CURRENT
                ):
                    return self._rejection(
                        "visibility",
                        "Belief update rejected: "
                        f"assertions.{index}.visibility must be SESSION_CURRENT when "
                        f"assertions.{index}.expiry_policy is END_OF_SESSION. Received "
                        f"visibility={item.visibility.value!r}, "
                        f"expiry_policy={item.expiry_policy.value!r}.",
                        repository_accessed=True,
                    )
                try:
                    subject = resolve_subject_reference(
                        item.subject_reference,
                        turn.user_text,
                        list(prepared.allowed_subjects),
                        turn.sender_id,
                    )
                except (TypeError, ValueError):
                    choices = list(prepared.grounded_subject_references())[:8]
                    received = item.subject_reference[:128]
                    return self._rejection(
                        "subject_reference_grounding",
                        "Belief update rejected: "
                        f"assertions.{index}.subject_reference must be copied exactly from "
                        "the current participant message and must be one of the frozen "
                        f"allowed references. Received: {received!r}. "
                        f"Choose one of: {choices!r}.",
                        repository_accessed=True,
                    )
                if item.evidence_excerpt not in turn.user_text:
                    received = item.evidence_excerpt[:160]
                    source = turn.user_text[:240]
                    return self._rejection(
                        "evidence_substring_validation",
                        "Belief update rejected: "
                        f"assertions.{index}.evidence_excerpt must be an exact substring "
                        f"of the current participant message. Received: {received!r}. "
                        f"Choose an exact excerpt from: {source!r}.",
                        repository_accessed=True,
                    )
                operations.append(AssertionCandidate(
                    operation="ASSERT",
                    subject_id=subject.subject_id,
                    **item.model_dump(),
                ))
            for item in wire.invalidations:
                if item.target_belief_id not in permitted_ids:
                    return self._rejection(
                        "frozen_catalog_validation",
                        "Belief update rejected: invalidation target is not in the frozen "
                        "permitted catalog for this participant message.",
                        repository_accessed=True,
                    )
                if item.evidence_excerpt not in turn.user_text:
                    received = item.evidence_excerpt[:160]
                    source = turn.user_text[:240]
                    return self._rejection(
                        "evidence_substring_validation",
                        "Belief update rejected: invalidations."
                        f"{len(operations) - len(wire.assertions)}.evidence_excerpt must be "
                        "an exact substring of the current participant message. "
                        f"Received: {received!r}. Choose an exact excerpt from: {source!r}.",
                        repository_accessed=True,
                    )
                operations.append(InvalidateCandidate(
                    operation="INVALIDATE",
                    **item.model_dump(),
                ))
            candidates = BeliefCandidateBatch(operations=operations)
            applied = self.update_service.apply(
                owner_agent_id=turn.owner_agent_id,
                session_id=turn.session_id,
                source_message_id=turn.user_message_id,
                user_text=turn.user_text,
                observed_at=turn.observed_at,
                timezone_name=turn.timezone_name,
                candidates=candidates,
                existing_beliefs=list(prepared.existing_beliefs),
                source_sender_id=turn.sender_id,
                source_sender_display_name=turn.sender_display_name,
                source_sender_type=turn.sender_type.value,
                source_input_source=turn.input_source.value,
                allowed_subjects=list(prepared.allowed_subjects),
            )
        except (TypeError, ValueError) as exc:
            return self._rejection(
                self._service_category(exc),
                f"Belief update rejected: {str(exc)[:700]}",
                repository_accessed=True,
            )
        if not applied:
            return ToolResult.success(
                "Belief changes for this participant message were already processed. "
                "Do not call beliefs__update again for this message."
            )
        return ToolResult.success(
            "Belief changes for this participant message were applied successfully "
            f"({len(wire.assertions)} assertion(s), {len(wire.invalidations)} invalidation(s)). "
            "Do not call beliefs__update again for this message."
        )

    @staticmethod
    def _rejection(category: str, reason: str, *, repository_accessed: bool) -> ToolResult:
        return ToolResult.error(reason, diagnostics={
            "category": category,
            "error_code": category.upper(),
            "repository_accessed": repository_accessed,
        })

    @staticmethod
    def _service_category(error: Exception) -> str:
        reason = str(error).casefold()
        if "evidence" in reason and "source" in reason:
            return "evidence_substring_validation"
        if "visibility" in reason:
            return "visibility"
        if "expiry" in reason or "explicit_until" in reason or "timezone" in reason:
            return "expiry_or_timestamp_validation"
        if "subject" in reason or "source sender" in reason:
            return "subject_authority"
        if "conflicting" in reason or "duplicate" in reason:
            return "candidate_normalization"
        return "service_rejection"

    @staticmethod
    def _validation_summary(error: ValidationError) -> str:
        parts = []
        for item in error.errors(include_url=False)[:6]:
            location = ".".join(str(part) for part in item.get("loc", ())) or "arguments"
            parts.append(f"{location}: {item.get('msg', 'invalid value')}")
        return "Belief update rejected: " + "; ".join(parts)[:700]

    def _schema(self) -> dict:
        expiry_values = [policy.value for policy in ExpiryPolicy]
        assertion = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "subject_reference", "predicate", "value", "visibility",
                "expiry_policy", "evidence_excerpt",
            ],
            "properties": {
                "subject_reference": {"type": "string", "minLength": 1, "maxLength": 128},
                "predicate": {"type": "string", "minLength": 2, "maxLength": 64},
                "value": {},
                "visibility": {
                    "type": "string",
                    "enum": [p.value for p in VisibilityPolicy],
                    "description": "Use SESSION_CURRENT when expiry_policy is END_OF_SESSION.",
                },
                "expiry_policy": {
                    "type": "string",
                    "enum": expiry_values,
                    "description": "END_OF_SESSION requires visibility SESSION_CURRENT.",
                },
                "explicit_until": {"type": ["string", "null"], "maxLength": 64},
                "evidence_excerpt": {"type": "string", "minLength": 1, "maxLength": 500},
            },
        }
        invalidation = {
            "type": "object",
            "additionalProperties": False,
            "required": ["target_belief_id", "evidence_excerpt"],
            "properties": {
                "target_belief_id": {"type": "string", "minLength": 1, "maxLength": 64},
                "evidence_excerpt": {"type": "string", "minLength": 1, "maxLength": 500},
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["assertions", "invalidations"],
            "properties": {
                "assertions": {"type": "array", "maxItems": self.max_candidates, "items": assertion},
                "invalidations": {"type": "array", "maxItems": self.max_candidates, "items": invalidation},
            },
            "anyOf": [
                {"properties": {"assertions": {"minItems": 1}}},
                {"properties": {"invalidations": {"minItems": 1}}},
            ],
        }
