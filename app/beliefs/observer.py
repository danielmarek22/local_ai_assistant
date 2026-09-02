from __future__ import annotations

import logging
import time

from app.beliefs.preparation import (
    BeliefTurnPreparer,
    is_conversational_belief_turn_eligible,
)
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
        max_allowed_subjects: int = 32,
        preparer=None,
    ):
        self.extractor = extractor
        self.update_service = update_service
        self.snapshot_service = snapshot_service
        self.history_store = history_store
        self.max_context_messages = max(0, int(max_context_messages))
        self.max_allowed_subjects = max(4, int(max_allowed_subjects))
        self.preparer = preparer or BeliefTurnPreparer(
            snapshot_service=snapshot_service,
            history_store=history_store,
            max_context_messages=self.max_context_messages,
            max_allowed_subjects=self.max_allowed_subjects,
        )

    def observe(self, completed_turn) -> None:
        if not is_conversational_belief_turn_eligible(completed_turn):
            return
        started = time.perf_counter()
        extraction_started = started
        extractor_called = False
        try:
            prepared = self.preparer.prepare(completed_turn)
            existing = list(prepared.existing_beliefs)
            context = list(prepared.disambiguating_context)
            allowed_subjects = list(prepared.allowed_subjects)
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

    _allowed_subjects = staticmethod(BeliefTurnPreparer._allowed_subjects)
