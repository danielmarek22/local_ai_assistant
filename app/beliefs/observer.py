from __future__ import annotations

import logging
import time

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
        started = time.perf_counter()
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
            {"role": row["role"], "content": row["content"]}
            for row in context
            if row["role"] in {"user", "assistant"}
        ]
        extraction_started = time.perf_counter()
        try:
            candidates = self.extractor.extract(
                user_text=completed_turn.user_text,
                disambiguating_context=context,
                existing_beliefs=existing,
                observed_at=completed_turn.observed_at,
                timezone_name=completed_turn.timezone_name,
            )
            extraction_duration_ms = (time.perf_counter() - extraction_started) * 1000
            applied = self.update_service.apply(
                owner_agent_id=completed_turn.owner_agent_id,
                session_id=completed_turn.session_id,
                source_message_id=completed_turn.user_message_id,
                user_text=completed_turn.user_text,
                observed_at=completed_turn.observed_at,
                timezone_name=completed_turn.timezone_name,
                candidates=candidates,
                existing_beliefs=existing,
            )
        except Exception as exc:
            total_duration_ms = (time.perf_counter() - started) * 1000
            extraction_duration_ms = (time.perf_counter() - extraction_started) * 1000
            logger.warning(
                "[%s] Belief extraction skipped after failure "
                "(model_duration=%.2f ms, total_duration=%.2f ms): %s",
                completed_turn.session_id,
                extraction_duration_ms,
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
                    "total_duration_ms": total_duration_ms,
                    "error": str(exc),
                },
            )
            return

        total_duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "[%s] Belief extraction complete (operations=%d, applied=%s, "
            "model_duration=%.2f ms, total_duration=%.2f ms)",
            completed_turn.session_id,
            len(candidates.operations),
            applied,
            extraction_duration_ms,
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
                "total_duration_ms": total_duration_ms,
            },
        )
