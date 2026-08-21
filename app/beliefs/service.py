from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from app.beliefs.models import (
    BeliefCandidateBatch,
    BeliefMutation,
    CandidateOperation,
    CreateCandidate,
    ExpiryPolicy,
    InvalidateCandidate,
    UpdateCandidate,
    VisibilityPolicy,
)
from app.beliefs.vocabulary import normalize_predicate


class BeliefUpdateService:
    def __init__(
        self,
        repository,
        *,
        extractor_version: str = "conversation-v1",
        max_value_chars: int = 2000,
        max_expiry_days: int = 90,
    ):
        self.repository = repository
        self.extractor_version = extractor_version
        self.max_value_chars = max(64, int(max_value_chars))
        self.max_expiry_days = max(1, int(max_expiry_days))

    def apply(
        self,
        *,
        owner_agent_id: str,
        session_id: str,
        source_message_id: int,
        user_text: str,
        observed_at: datetime,
        timezone_name: str,
        candidates: BeliefCandidateBatch,
        existing_beliefs: list,
    ) -> bool:
        if self.repository.has_application(
            owner_agent_id, source_message_id, self.extractor_version
        ):
            return False

        existing_by_id = {belief.belief_id: belief for belief in existing_beliefs}
        logical_keys = {
            (
                belief.visibility,
                belief.origin_session_id if belief.visibility == VisibilityPolicy.SESSION_CURRENT else "",
                belief.subject,
                belief.predicate,
            ): belief
            for belief in existing_beliefs
        }
        mutations: list[BeliefMutation] = []

        for candidate in candidates.operations:
            if candidate.operation == CandidateOperation.IGNORE:
                continue
            evidence_excerpt = getattr(candidate, "evidence_excerpt", None)
            if evidence_excerpt is not None and evidence_excerpt not in user_text:
                raise ValueError("Belief evidence excerpt is not present in the source user message")

            if isinstance(candidate, CreateCandidate):
                subject = self._normalize_subject(candidate.subject)
                predicate = normalize_predicate(candidate.predicate)
                scope_session = (
                    session_id if candidate.visibility == VisibilityPolicy.SESSION_CURRENT else ""
                )
                key = (candidate.visibility, scope_session, subject, predicate)
                if key in logical_keys:
                    raise ValueError(
                        "CREATE cannot replace an existing semantic property; use UPDATE"
                    )
                self._validate_value(candidate.value)
                mutations.append(BeliefMutation(
                    operation=CandidateOperation.CREATE,
                    belief_id=None,
                    visibility=candidate.visibility,
                    origin_session_id=session_id,
                    subject=subject,
                    predicate=predicate,
                    value=candidate.value,
                    expires_at=self._resolve_expiry(
                        candidate.expiry_policy,
                        candidate.explicit_until,
                        candidate.visibility,
                        observed_at,
                        timezone_name,
                    ),
                    evidence_excerpt=evidence_excerpt,
                ))
                continue

            target = existing_by_id.get(candidate.target_belief_id)
            if target is None or target.owner_agent_id != owner_agent_id:
                raise ValueError("Belief target was not supplied as an active belief")
            if (
                target.visibility == VisibilityPolicy.SESSION_CURRENT
                and target.origin_session_id != session_id
            ):
                raise ValueError("Session-scoped belief belongs to a different session")

            if isinstance(candidate, UpdateCandidate):
                self._validate_value(candidate.value)
                mutations.append(BeliefMutation(
                    operation=CandidateOperation.UPDATE,
                    belief_id=target.belief_id,
                    visibility=target.visibility,
                    origin_session_id=session_id,
                    subject=target.subject,
                    predicate=target.predicate,
                    value=candidate.value,
                    expires_at=self._resolve_expiry(
                        candidate.expiry_policy,
                        candidate.explicit_until,
                        target.visibility,
                        observed_at,
                        timezone_name,
                    ),
                    evidence_excerpt=evidence_excerpt,
                ))
                continue

            if isinstance(candidate, InvalidateCandidate):
                mutations.append(BeliefMutation(
                    operation=CandidateOperation.INVALIDATE,
                    belief_id=target.belief_id,
                    visibility=None,
                    origin_session_id=session_id,
                    subject=target.subject,
                    predicate=target.predicate,
                    evidence_excerpt=evidence_excerpt,
                ))

        return self.repository.apply_mutations(
            owner_agent_id=owner_agent_id,
            source_message_id=source_message_id,
            extractor_version=self.extractor_version,
            mutations=mutations,
            now=observed_at,
        )

    def _resolve_expiry(
        self,
        policy: ExpiryPolicy,
        explicit_until: str | None,
        visibility: VisibilityPolicy,
        observed_at: datetime,
        timezone_name: str,
    ) -> datetime | None:
        if observed_at.tzinfo is None:
            raise ValueError("Completed-turn timestamp must be timezone-aware")
        observed_at = observed_at.astimezone(timezone.utc)
        if policy == ExpiryPolicy.END_OF_SESSION:
            if visibility != VisibilityPolicy.SESSION_CURRENT:
                raise ValueError("END_OF_SESSION requires SESSION_CURRENT visibility")
            if explicit_until is not None:
                raise ValueError("END_OF_SESSION does not accept explicit_until")
            return None
        if policy == ExpiryPolicy.NO_AUTOMATIC_EXPIRY:
            if explicit_until is not None:
                raise ValueError("NO_AUTOMATIC_EXPIRY does not accept explicit_until")
            return None
        if policy == ExpiryPolicy.AFTER_ONE_HOUR:
            expiry = observed_at + timedelta(hours=1)
        elif policy == ExpiryPolicy.AFTER_TWENTY_FOUR_HOURS:
            expiry = observed_at + timedelta(hours=24)
        elif policy == ExpiryPolicy.AFTER_SEVEN_DAYS:
            expiry = observed_at + timedelta(days=7)
        elif policy == ExpiryPolicy.END_OF_LOCAL_DAY:
            local_zone = ZoneInfo(timezone_name)
            local = observed_at.astimezone(local_zone)
            expiry = datetime.combine(
                local.date() + timedelta(days=1),
                time.min,
                tzinfo=local_zone,
            ).astimezone(timezone.utc)
        elif policy == ExpiryPolicy.UNTIL_EXPLICIT_DATETIME:
            if not explicit_until:
                raise ValueError("UNTIL_EXPLICIT_DATETIME requires explicit_until")
            expiry = datetime.fromisoformat(explicit_until.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                raise ValueError("explicit_until must include a timezone")
            expiry = expiry.astimezone(timezone.utc)
        else:
            raise ValueError(f"Unsupported expiry policy: {policy}")

        if explicit_until is not None and policy != ExpiryPolicy.UNTIL_EXPLICIT_DATETIME:
            raise ValueError(f"{policy.value} does not accept explicit_until")
        if expiry <= observed_at:
            raise ValueError("Belief expiry must be after the source message")
        if expiry > observed_at + timedelta(days=self.max_expiry_days):
            raise ValueError("Belief expiry exceeds the configured maximum horizon")
        return expiry

    def _validate_value(self, value) -> None:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True)
        if len(rendered) > self.max_value_chars:
            raise ValueError("Belief value exceeds the configured size limit")

    @staticmethod
    def _normalize_subject(value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"user", "world", "environment"}:
            raise ValueError("Conversational belief subject must be user, world, or environment")
        return normalized
