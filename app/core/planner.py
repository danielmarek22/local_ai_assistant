import re


class PlannerDecision:
    def __init__(self, use_web: bool, query: str | None = None):
        self.use_web = use_web
        self.query = query


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
                    use_web=True,
                    query=user_text
                )

        print(PlannerDecision)
        return PlannerDecision(use_web=False)
