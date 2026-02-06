import re
import logging

from app.core.actions import Action
from app.core.plan import Plan
from app.core.intents import is_memory_command, extract_memory_content

logger = logging.getLogger("planner")


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
        logger.info("Planner invoked (len=%d)", len(user_text))
        logger.debug("Planner input text: %r", user_text)

        actions = []
        text = user_text.lower().strip()

        # --------------------------------------------------
        # 1. Explicit memory command
        # --------------------------------------------------
        if is_memory_command(user_text):
            logger.info("Memory command detected")

            content = extract_memory_content(user_text)

            if content:
                logger.debug(
                    "Extracted memory content (%d chars)",
                    len(content),
                )
                actions.append(
                    Action(
                        type="write_memory",
                        payload={"content": content},
                    )
                )
            else:
                logger.warning("Memory command detected but no content extracted")

            actions.append(Action(type="respond"))

            logger.info(
                "Planner decision: memory_write + respond (%d actions)",
                len(actions),
            )
            return Plan(actions=actions)

        # --------------------------------------------------
        # 2. Web search intent
        # --------------------------------------------------
        for pattern in self.SEARCH_PATTERNS:
            if re.search(pattern, text):
                logger.info(
                    "Web search intent detected (pattern=%r)",
                    pattern,
                )
                plan = Plan(
                    actions=[
                        Action(
                            type="web_search",
                            payload={"query": user_text},
                        ),
                        Action(type="respond"),
                    ]
                )

                logger.info(
                    "Planner decision: web_search + respond (%d actions)",
                    len(plan.actions),
                )
                return plan

        # --------------------------------------------------
        # 3. Default response
        # --------------------------------------------------
        logger.info("Planner decision: default respond")

        return Plan(
            actions=[
                Action(type="respond"),
            ]
        )
