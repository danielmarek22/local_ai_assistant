import re
from app.core.actions import Action
from app.core.plan import Plan
from app.core.intents import is_memory_command, extract_memory_content
from app.core.actions import Action
from app.core.plan import Plan



import re
from app.core.actions import Action
from app.core.plan import Plan
from app.core.intents import is_memory_command, extract_memory_content


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

    def decide(self, user_text: str) -> Plan:
        actions = []
        text = user_text.lower().strip()

        # --------------------------------------------------
        # 1. Explicit memory command
        # --------------------------------------------------
        if is_memory_command(user_text):
            content = extract_memory_content(user_text)

            if content:
                actions.append(
                    Action(
                        type="write_memory",
                        payload={"content": content},
                    )
                )

            # Always respond after memory intent
            actions.append(Action(type="respond"))
            return Plan(actions=actions)

        # --------------------------------------------------
        # 2. Web search intent
        # --------------------------------------------------
        for pattern in self.SEARCH_PATTERNS:
            if re.search(pattern, text):
                return Plan(
                    actions=[
                        Action(
                            type="web_search",
                            payload={"query": user_text},
                        ),
                        Action(type="respond"),
                    ]
                )

        # --------------------------------------------------
        # 3. Default response
        # --------------------------------------------------
        return Plan(
            actions=[
                Action(type="respond"),
            ]
        )

