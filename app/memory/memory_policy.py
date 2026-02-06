from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryDecision:
    content: str
    category: str = "general"
    importance: int = 1


class SimpleMemoryPolicy:
    """
    Translates planner memory actions into concrete storage decisions.
    """

    def decide_from_action(self, action_payload: dict) -> Optional[MemoryDecision]:
        content = action_payload.get("content")
        if not content:
            return None

        # You can evolve this logic later
        return MemoryDecision(
            content=content,
            category=action_payload.get("category", "general"),
            importance=action_payload.get("importance", 2),
        )
