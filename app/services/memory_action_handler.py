import logging

from app.core.actions import Action
from app.logging import trace_event


logger = logging.getLogger("memory_action_handler")


class MemoryActionHandler:
    def __init__(self, memory_store, memory_policy):
        self.memory = memory_store
        self.memory_policy = memory_policy

    def handle(self, session_id: str, action: Action) -> None:
        self.handle_payload(session_id, action.payload or {})

    def handle_payload(self, session_id: str, payload: dict) -> bool:
        logger.debug("[%s] Processing memory action", session_id)

        decision = self.memory_policy.decide_from_action(payload)

        if not decision:
            logger.debug("[%s] Memory action ignored by policy", session_id)
            trace_event(
                "memory_action",
                "memory_action_skipped",
                session_id=session_id,
                payload={"action_payload": payload},
            )
            return False

        trace_event(
            "memory_action",
            "memory_action_applied",
            session_id=session_id,
            payload={
                "action_payload": payload,
                "decision": {
                    "content": decision.content,
                    "category": decision.category,
                    "importance": decision.importance,
                },
            },
        )

        self.memory.add(
            content=decision.content,
            category=decision.category,
            importance=decision.importance,
        )

        logger.info(
            "[%s] Memory written (category=%s, importance=%d)",
            session_id,
            decision.category,
            decision.importance,
        )
        return True
