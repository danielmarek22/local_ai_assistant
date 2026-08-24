from __future__ import annotations

import logging
import time

from app.beliefs.models import AllowedSubject
from app.beliefs.subjects import default_allowed_subjects, participant_subject, subject_from_belief
from app.core.conversation import InputSource, SenderType
from app.logging import trace_event


logger = logging.getLogger("belief_observer")


class ConversationalBeliefObserver:
    def __init__(
        self,
        *,
        extractor,
        update_service,
        snapshot_service,
        history_store,
        max_context_messages: int = 2,
    ):
        self.extractor = extractor
        self.update_service = update_service
        self.snapshot_service = snapshot_service
        self.history_store = history_store
        self.max_context_messages = max(0, int(max_context_messages))

    def observe(self, completed_turn) -> None:
        if not completed_turn.user_text.strip():
            return
        if completed_turn.sender_type not in {SenderType.HUMAN, SenderType.EXTERNAL_AGENT}:
            return
        if completed_turn.input_source not in {
            InputSource.LOCAL_TEXT,
            InputSource.LOCAL_VOICE,
            InputSource.MANUAL_RELAY,
        }:
            return
        started = time.perf_counter()
        extraction_started = started
        extractor_called = False
        try:
            existing = self.snapshot_service.relevant_for_extraction(
                completed_turn.owner_agent_id,
                completed_turn.session_id,
                completed_turn.user_text,
                now=completed_turn.observed_at,
            )
            context = self.history_store.get_before(
                completed_turn.session_id,
                completed_turn.user_message_id,
                limit=self.max_context_messages,
            )
            context = [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "sender_id": row.get("sender_id", ""),
                    "sender_display_name": row.get("sender_display_name", ""),
                    "sender_type": row.get("sender_type", ""),
                }
                for row in context
                if row["role"] in {"user", "assistant"}
            ]
            allowed_subjects = self._allowed_subjects(completed_turn, context, existing)
            extraction_started = time.perf_counter()
            extractor_called = True
            candidates = self.extractor.extract(
                user_text=completed_turn.user_text,
                disambiguating_context=context,
                existing_beliefs=existing,
                allowed_subjects=allowed_subjects,
                source_sender_id=completed_turn.sender_id,
                source_sender_display_name=completed_turn.sender_display_name,
                source_sender_type=completed_turn.sender_type.value,
                observed_at=completed_turn.observed_at,
                timezone_name=completed_turn.timezone_name,
            )
            extraction_duration_ms = (time.perf_counter() - extraction_started) * 1000
            attempt_count = getattr(self.extractor, "last_attempt_count", 1)
            model_duration_ms = getattr(
                self.extractor, "last_model_duration_ms", extraction_duration_ms
            )
            applied = self.update_service.apply(
                owner_agent_id=completed_turn.owner_agent_id,
                session_id=completed_turn.session_id,
                source_message_id=completed_turn.user_message_id,
                user_text=completed_turn.user_text,
                observed_at=completed_turn.observed_at,
                timezone_name=completed_turn.timezone_name,
                candidates=candidates,
                existing_beliefs=existing,
                source_sender_id=completed_turn.sender_id,
                source_sender_display_name=completed_turn.sender_display_name,
                source_sender_type=completed_turn.sender_type.value,
                source_input_source=completed_turn.input_source.value,
                allowed_subjects=allowed_subjects,
            )
        except Exception as exc:
            total_duration_ms = (time.perf_counter() - started) * 1000
            extraction_duration_ms = (time.perf_counter() - extraction_started) * 1000
            attempt_count = (
                getattr(self.extractor, "last_attempt_count", 1)
                if extractor_called else 0
            )
            model_duration_ms = (
                getattr(self.extractor, "last_model_duration_ms", extraction_duration_ms)
                if extractor_called else 0.0
            )
            logger.warning(
                "[%s] Belief extraction skipped after failure "
                "(attempts=%d, model_duration=%.2f ms, total_duration=%.2f ms): %s",
                completed_turn.session_id,
                attempt_count,
                model_duration_ms,
                total_duration_ms,
                exc,
            )
            trace_event(
                "belief_observer",
                "extraction_failed",
                session_id=completed_turn.session_id,
                payload={
                    "source_message_id": completed_turn.user_message_id,
                    "extraction_duration_ms": extraction_duration_ms,
                    "model_duration_ms": model_duration_ms,
                    "attempt_count": attempt_count,
                    "total_duration_ms": total_duration_ms,
                    "error": str(exc),
                },
            )
            return

        total_duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "[%s] Belief extraction complete (operations=%d, applied=%s, "
            "attempts=%d, model_duration=%.2f ms, total_duration=%.2f ms)",
            completed_turn.session_id,
            len(candidates.operations),
            applied,
            attempt_count,
            model_duration_ms,
            total_duration_ms,
        )
        trace_event(
            "belief_observer",
            "extraction_complete",
            session_id=completed_turn.session_id,
            payload={
                "source_message_id": completed_turn.user_message_id,
                "operation_count": len(candidates.operations),
                "applied": applied,
                "extraction_duration_ms": extraction_duration_ms,
                "model_duration_ms": model_duration_ms,
                "attempt_count": attempt_count,
                "total_duration_ms": total_duration_ms,
            },
        )

    @staticmethod
    def _allowed_subjects(completed_turn, context, existing) -> list[AllowedSubject]:
        ordered = default_allowed_subjects(
            completed_turn.sender_id,
            completed_turn.sender_display_name,
            completed_turn.sender_type.value,
        )
        ordered.extend(subject_from_belief(belief) for belief in existing)
        for item in context:
            if item.get("sender_type") not in {
                SenderType.HUMAN.value,
                SenderType.EXTERNAL_AGENT.value,
            }:
                continue
            sender_id = item.get("sender_id")
            display_name = item.get("sender_display_name")
            if sender_id and display_name:
                ordered.append(participant_subject(
                    sender_id,
                    display_name,
                    item["sender_type"],
                ))
        resolved = {}
        for subject in ordered:
            previous = resolved.get(subject.subject_id)
            if previous is not None and previous.subject_kind != subject.subject_kind:
                raise ValueError("Ambiguous authoritative subject metadata")
            if previous is None:
                resolved[subject.subject_id] = subject
        return list(resolved.values())
