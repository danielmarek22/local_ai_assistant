from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from app.beliefs.models import (
    AllowedSubject,
    AssertionCandidate,
    BeliefCandidateBatch,
    BeliefMutation,
    CandidateOperation,
    CreateCandidate,
    EpistemicStatus,
    ExpiryPolicy,
    InvalidateCandidate,
    UpdateCandidate,
    VisibilityPolicy,
)
from app.beliefs.vocabulary import normalize_predicate
from app.beliefs.version import CONVERSATIONAL_BELIEF_EXTRACTOR_VERSION


APPLICATION_OWNED_EXPIRY_POLICIES = {
    "current_activity": ExpiryPolicy.END_OF_LOCAL_DAY,
}

logger = logging.getLogger("belief_update_service")


class BeliefUpdateService:
    def __init__(
        self,
        repository,
        *,
        extractor_version: str = CONVERSATIONAL_BELIEF_EXTRACTOR_VERSION,
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
        source_sender_id: str = "local-human",
        source_sender_display_name: str = "You",
        source_sender_type: str = "human",
        source_input_source: str = "local_text",
        allowed_subjects: list[AllowedSubject] | None = None,
    ) -> bool:
        if self.repository.has_application(
            owner_agent_id, source_message_id, self.extractor_version
        ):
            return False

        operation_counts = Counter(
            candidate.operation.value for candidate in candidates.operations
        )
        logger.debug(
            "Belief batch preflight started operation_counts=%s",
            json.dumps(dict(sorted(operation_counts.items())), sort_keys=True),
        )

        allowed_by_id = self._allowed_subjects(
            allowed_subjects,
            existing_beliefs,
            source_sender_id=source_sender_id,
            source_sender_display_name=source_sender_display_name,
            source_sender_type=source_sender_type,
        )
        existing_by_id = {belief.belief_id: belief for belief in existing_beliefs}
        logical_keys = {
            (
                belief.visibility,
                belief.source_session_id if belief.visibility == VisibilityPolicy.SESSION_CURRENT else "",
                belief.subject_id,
                belief.predicate,
                belief.epistemic_status,
                belief.source_sender_id,
            ): belief
            for belief in existing_beliefs
        }
        mutations: list[BeliefMutation] = []
        assertions_by_key: dict[tuple, BeliefMutation] = {}
        assertions_by_property: dict[tuple, BeliefMutation] = {}
        invalidations_by_id: dict[str, BeliefMutation] = {}

        for candidate in candidates.operations:
            if candidate.operation == CandidateOperation.IGNORE:
                continue
            evidence_excerpt = getattr(candidate, "evidence_excerpt", None)
            if not evidence_excerpt:
                raise ValueError("Conversational belief mutation requires evidence_excerpt")
            if evidence_excerpt not in user_text:
                raise ValueError("Belief evidence excerpt is not present in the source user message")

            if isinstance(candidate, (AssertionCandidate, CreateCandidate)):
                subject = allowed_by_id.get(candidate.subject_id)
                if subject is None:
                    raise ValueError("Belief subject_id was not supplied as an allowed subject")
                epistemic_status = (
                    EpistemicStatus.SELF_REPORT
                    if subject.subject_id == source_sender_id
                    else EpistemicStatus.ATTRIBUTED_CLAIM
                )
                predicate = normalize_predicate(candidate.predicate)
                scope_session = (
                    session_id if candidate.visibility == VisibilityPolicy.SESSION_CURRENT else ""
                )
                key = (
                    candidate.visibility,
                    scope_session,
                    subject.subject_id,
                    predicate,
                    epistemic_status,
                    source_sender_id,
                )
                if isinstance(candidate, CreateCandidate) and key in logical_keys:
                    raise ValueError(
                        "CREATE cannot replace an existing semantic property; use UPDATE"
                    )
                self._validate_value(candidate.value)
                mutation = BeliefMutation(
                    operation=(
                        CandidateOperation.ASSERT
                        if isinstance(candidate, AssertionCandidate)
                        else CandidateOperation.CREATE
                    ),
                    belief_id=None,
                    visibility=candidate.visibility,
                    source_session_id=session_id,
                    subject_id=subject.subject_id,
                    subject_kind=subject.subject_kind,
                    subject_display_name=subject.subject_display_name,
                    predicate=predicate,
                    epistemic_status=epistemic_status,
                    source_sender_id=source_sender_id,
                    source_sender_display_name=source_sender_display_name,
                    source_sender_type=source_sender_type,
                    source_input_source=source_input_source,
                    value=candidate.value,
                    expires_at=self._resolve_assertion_expiry(
                        predicate,
                        candidate.expiry_policy,
                        candidate.explicit_until,
                        candidate.visibility,
                        observed_at,
                        timezone_name,
                    ),
                    evidence_excerpt=evidence_excerpt,
                )
                property_key = (
                    subject.subject_id,
                    predicate,
                    epistemic_status,
                    source_sender_id,
                )
                property_assertion = assertions_by_property.get(property_key)
                if property_assertion is not None and property_assertion != mutation:
                    logger.debug(
                        "Belief batch conflict category=CONFLICTING_ASSERTIONS "
                        "track_fingerprint=%s",
                        self._logical_key_fingerprint(key),
                    )
                    raise ValueError(
                        "Batch contains conflicting operations for the same belief track"
                    )
                previous = assertions_by_key.get(key)
                if previous is not None:
                    if previous == mutation:
                        logger.debug(
                            "Belief batch normalization category=IDENTICAL_DUPLICATE_ASSERTION "
                            "track_fingerprint=%s",
                            self._logical_key_fingerprint(key),
                        )
                        continue
                    logger.debug(
                        "Belief batch conflict category=CONFLICTING_ASSERTIONS "
                        "track_fingerprint=%s",
                        self._logical_key_fingerprint(key),
                    )
                    raise ValueError(
                        "Batch contains conflicting operations for the same belief track"
                    )
                assertions_by_property[property_key] = mutation
                assertions_by_key[key] = mutation
                mutations.append(mutation)
                continue

            target = existing_by_id.get(candidate.target_belief_id)
            if target is None or target.owner_agent_id != owner_agent_id:
                raise ValueError("Belief target was not supplied as an active belief")
            if target.source_sender_id != source_sender_id:
                raise ValueError("Belief target belongs to a different evidence source")
            expected_status = (
                EpistemicStatus.SELF_REPORT
                if target.subject_id == source_sender_id
                else EpistemicStatus.ATTRIBUTED_CLAIM
            )
            if target.epistemic_status != expected_status:
                raise ValueError("Belief target has inconsistent epistemic attribution")
            if (
                target.visibility == VisibilityPolicy.SESSION_CURRENT
                and target.source_session_id != session_id
            ):
                raise ValueError("Session-scoped belief belongs to a different session")

            if isinstance(candidate, UpdateCandidate):
                self._validate_value(candidate.value)
                resolved_subject = allowed_by_id[target.subject_id]
                mutations.append(BeliefMutation(
                    operation=CandidateOperation.UPDATE,
                    belief_id=target.belief_id,
                    visibility=target.visibility,
                    source_session_id=session_id,
                    subject_id=target.subject_id,
                    subject_kind=resolved_subject.subject_kind,
                    subject_display_name=resolved_subject.subject_display_name,
                    predicate=target.predicate,
                    epistemic_status=target.epistemic_status,
                    source_sender_id=source_sender_id,
                    source_sender_display_name=source_sender_display_name,
                    source_sender_type=source_sender_type,
                    source_input_source=source_input_source,
                    value=candidate.value,
                    expires_at=self._resolve_assertion_expiry(
                        target.predicate,
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
                if target.belief_id in invalidations_by_id:
                    logger.debug(
                        "Belief batch conflict category=DUPLICATE_INVALIDATION "
                        "track_fingerprint=%s",
                        self._logical_key_fingerprint(self._logical_key_for_belief(target)),
                    )
                    raise ValueError(
                        "Batch contains conflicting operations for the same belief track"
                    )
                mutation = BeliefMutation(
                    operation=CandidateOperation.INVALIDATE,
                    belief_id=target.belief_id,
                    visibility=None,
                    source_session_id=session_id,
                    subject_id=target.subject_id,
                    subject_kind=target.subject_kind,
                    subject_display_name=target.subject_display_name,
                    predicate=target.predicate,
                    epistemic_status=target.epistemic_status,
                    source_sender_id=source_sender_id,
                    source_sender_display_name=source_sender_display_name,
                    source_sender_type=source_sender_type,
                    source_input_source=source_input_source,
                    evidence_excerpt=evidence_excerpt,
                )
                invalidations_by_id[target.belief_id] = mutation
                mutations.append(mutation)

        invalidated_keys = {
            belief_id: self._logical_key_for_belief(existing_by_id[belief_id])
            for belief_id in invalidations_by_id
        }
        redundant_invalidation_ids = {
            belief_id
            for belief_id, key in invalidated_keys.items()
            if key in assertions_by_key
        }
        if redundant_invalidation_ids:
            for belief_id in sorted(redundant_invalidation_ids):
                logger.debug(
                    "Belief batch normalization category=REDUNDANT_ASSERT_INVALIDATE "
                    "track_fingerprint=%s",
                    self._logical_key_fingerprint(invalidated_keys[belief_id]),
                )
            mutations = [
                mutation
                for mutation in mutations
                if not (
                    mutation.operation == CandidateOperation.INVALIDATE
                    and mutation.belief_id in redundant_invalidation_ids
                )
            ]

        logger.debug(
            "Belief batch preflight complete normalized_counts=%s "
            "assertion_track_fingerprints=%s invalidation_track_fingerprints=%s",
            json.dumps(dict(sorted(Counter(
                mutation.operation.value for mutation in mutations
            ).items())), sort_keys=True),
            sorted(self._logical_key_fingerprint(key) for key in assertions_by_key),
            sorted(
                self._logical_key_fingerprint(key)
                for belief_id, key in invalidated_keys.items()
                if belief_id not in redundant_invalidation_ids
            ),
        )

        return self.repository.apply_mutations(
            owner_agent_id=owner_agent_id,
            source_message_id=source_message_id,
            extractor_version=self.extractor_version,
            mutations=mutations,
            now=observed_at,
        )

    def _resolve_assertion_expiry(
        self,
        predicate: str,
        requested_policy: ExpiryPolicy,
        explicit_until: str | None,
        visibility: VisibilityPolicy,
        observed_at: datetime,
        timezone_name: str,
    ) -> datetime | None:
        application_policy = APPLICATION_OWNED_EXPIRY_POLICIES.get(predicate)
        if application_policy is not None:
            if explicit_until is not None:
                raise ValueError(
                    f"{predicate} uses application-owned {application_policy.value} expiry"
                )
            requested_policy = application_policy
        return self._resolve_expiry(
            requested_policy,
            explicit_until,
            visibility,
            observed_at,
            timezone_name,
        )

    @staticmethod
    def _logical_key_for_belief(belief) -> tuple:
        return (
            belief.visibility,
            (
                belief.source_session_id
                if belief.visibility == VisibilityPolicy.SESSION_CURRENT
                else ""
            ),
            belief.subject_id,
            belief.predicate,
            belief.epistemic_status,
            belief.source_sender_id,
        )

    @staticmethod
    def _logical_key_fingerprint(key: tuple) -> str:
        normalized = [getattr(part, "value", part) for part in key]
        encoded = json.dumps(
            normalized,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

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
    def _allowed_subjects(
        allowed_subjects,
        existing_beliefs,
        *,
        source_sender_id,
        source_sender_display_name,
        source_sender_type,
    ):
        from app.beliefs.subjects import default_allowed_subjects, subject_from_belief

        subjects = list(allowed_subjects or default_allowed_subjects(
            source_sender_id,
            source_sender_display_name,
            source_sender_type,
        ))
        subjects.extend(subject_from_belief(belief) for belief in existing_beliefs)
        resolved = {}
        for subject in subjects:
            previous = resolved.get(subject.subject_id)
            if previous is not None and previous.subject_kind != subject.subject_kind:
                raise ValueError("Ambiguous allowed subject metadata")
            if previous is None:
                resolved[subject.subject_id] = subject
        return resolved
