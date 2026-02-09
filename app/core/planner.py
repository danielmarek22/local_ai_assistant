import re
import logging
from typing import Dict, Optional

from app.core.actions import Action
from app.core.plan import Plan

logger = logging.getLogger("planner")


class Planner:
    # ============================================================
    # Pattern definitions (rule-based heuristics)
    # ============================================================

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

    MEMORY_PATTERNS = [
        r"\bremember that\b",
        r"\bremember this\b",
        r"\bplease remember\b",
        r"\bnote that\b",
        r"\bsave this\b",
    ]

    # ============================================================
    # Public API
    # ============================================================

    def decide(self, user_text: str, perception: dict) -> Plan:
        logger.info("Planner invoked (len=%d)", len(user_text))
        logger.debug("Planner input text: %r", user_text)

        logger.debug(
            "Planner perception keys: %s",
            list(perception.keys()),
        )

        text = user_text.lower().strip()

        # --------------------------------------------------
        # 1. Explicit memory command
        # --------------------------------------------------
        memory_content = self._extract_memory_content(user_text)
        if memory_content is not None:
            logger.info("Memory command detected")

            actions = [
                Action(
                    type="write_memory",
                    payload={"content": memory_content},
                ),
                Action(type="respond"),
            ]

            logger.info(
                "Planner decision: memory_write + respond (%d actions)",
                len(actions),
            )
            return Plan(actions=actions)

        # --------------------------------------------------
        # 2. Web search intent
        # --------------------------------------------------
        if self._is_search_query(text):
            logger.info("Web search intent detected")

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
        logger.info("Planner decision: default respond")
        return Plan(actions=[Action(type="respond")])

    # ============================================================
    # Intent detection helpers (private)
    # ============================================================

    def _is_search_query(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in self.SEARCH_PATTERNS)

    def _extract_memory_content(self, text: str) -> Optional[str]:
        """
        If the text is a memory command, return the extracted content.
        Otherwise return None.
        """
        lowered = text.lower()

        for pattern in self.MEMORY_PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                start = match.end()
                content = text[start:].strip(" :.-")

                if content:
                    logger.debug(
                        "Extracted memory content (%d chars)",
                        len(content),
                    )
                    return content

                logger.warning("Memory command detected but no content extracted")
                return ""

        return None
