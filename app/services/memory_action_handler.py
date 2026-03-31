import logging

from app.core.actions import Action


logger = logging.getLogger("memory_action_handler")


class MemoryActionHandler:
    def __init__(self, memory_store, memory_policy):
        self.memory = memory_store
        self.memory_policy = memory_policy

    def handle(self, session_id: str, action: Action) -> None:
        logger.debug("[%s] Processing memory action", session_id)

        decision = self.memory_policy.decide_from_action(action.payload or {})

        if not decision:
            logger.debug("[%s] Memory action ignored by policy", session_id)
            return

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
