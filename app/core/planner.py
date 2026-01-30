import re

class PlannerDecision:
    def __init__(
        self,
        action: str,              # "respond" | "web_search"
        query: str | None = None, # rewritten query
        reason: str | None = None # optional, for debugging
    ):
        self.action = action
        self.query = query
        self.reason = reason



class Planner:
    SEARCH_PATTERNS = [
        r"\blatest\b",
        r"\bcurrent\b",
        r"\btoday\b",
        r"\bnews\b",
        r"^what is\b",
        r"^who is\b",
        r"^when did\b",
        r"^where is\b",
    ]

    def decide(self, user_text: str) -> PlannerDecision:
        text = user_text.lower().strip()

        for pattern in self.SEARCH_PATTERNS:
            if re.search(pattern, text):
                return PlannerDecision(
                    action="web_search",
                    query=user_text,
                    reason="rule_match"
                )

        return PlannerDecision(
            action="respond",
            reason="rule_no_match"
        )
