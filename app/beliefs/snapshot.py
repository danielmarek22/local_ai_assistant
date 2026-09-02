from __future__ import annotations

from datetime import datetime


class BeliefSnapshotService:
    def __init__(self, repository, *, max_beliefs: int = 24):
        self.repository = repository
        self.max_beliefs = max(1, int(max_beliefs))

    def active_for_turn(
        self,
        owner_agent_id: str,
        session_id: str,
        *,
        now: datetime | None = None,
    ):
        beliefs = self.repository.get_visible(
            owner_agent_id,
            session_id,
            now=now,
        )
        effective = {}
        for belief in beliefs:
            key = (
                belief.subject_id,
                belief.predicate,
                belief.epistemic_status,
                belief.source_sender_id,
            )
            current = effective.get(key)
            if current is None or belief.visibility.value == "SESSION_CURRENT":
                effective[key] = belief
        return sorted(
            effective.values(),
            key=lambda belief: (
                belief.subject_id,
                belief.predicate,
                belief.epistemic_status.value,
                belief.source_sender_id,
                belief.visibility.value,
                belief.belief_id,
            ),
        )[: self.max_beliefs]

    def visible_for_extraction(
        self,
        owner_agent_id: str,
        session_id: str,
        *,
        now: datetime | None = None,
    ):
        """Return underlying visible rows without scope collapsing or snapshot limits."""
        return self.repository.get_visible(
            owner_agent_id,
            session_id,
            now=now,
        )

    def relevant_for_extraction(
        self,
        owner_agent_id: str,
        session_id: str,
        user_text: str,
        *,
        now: datetime | None = None,
    ):
        beliefs = self.visible_for_extraction(owner_agent_id, session_id, now=now)
        terms = {term for term in user_text.lower().split() if len(term) >= 4}
        if not terms:
            return beliefs
        ranked = sorted(
            beliefs,
            key=lambda belief: (
                -sum(
                    term in (
                        f"{belief.subject_id} {belief.subject_display_name} "
                        f"{belief.source_sender_display_name} {belief.predicate} {belief.value}"
                    ).lower()
                    for term in terms
                ),
                belief.subject_id,
                belief.predicate,
                belief.epistemic_status.value,
                belief.source_sender_id,
                belief.belief_id,
            ),
        )
        return ranked
