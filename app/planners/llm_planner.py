import json
import time
import logging
import re
from typing import Dict, Optional

from app.core.actions import Action
from app.core.plan import Plan

logger = logging.getLogger("llm_planner")


class LLMPlanner:
    def __init__(self, llm, timeout_ms: int = 1500):
        self.llm = llm
        self.timeout_ms = timeout_ms

        logger.info(
            "LLMPlanner initialized (timeout_ms=%d)",
            timeout_ms,
        )

    def decide(self, user_text: str, perception: dict) -> Plan:
        start_ts = time.perf_counter()

        logger.info(
            "LLMPlanner invoked (len=%d)",
            len(user_text),
        )
        logger.debug("LLMPlanner input text: %r", user_text)

        perception_text = self._format_perception(perception)

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a planner for an AI assistant.\n"
                    "Decide what actions to take.\n"
                    "Use the current environment context if helpful.\n"
                    "Output ONLY valid JSON.\n\n"
                    "Current perception:\n"
                    f"{perception_text}\n\n"
                    "Schema:\n"
                    "{\n"
                    '  "actions": [\n'
                    '    { "type": "web_search", "query": string } | '
                    '{ "type": "respond" } |\n'
                    '    { "type": "write_memory", "content": string }\n'
                    "  ]\n"
                    "}"
                ),
            },
            {"role": "user", "content": user_text},
        ]

        buffer = ""
        timed_out = False

        for chunk in self.llm.stream_chat(prompt):
            buffer += chunk

            if (time.perf_counter() - start_ts) * 1000 > self.timeout_ms:
                timed_out = True
                logger.warning("LLMPlanner timeout")
                break

        logger.debug("LLMPlanner raw output: %r", buffer)

        try:
            data = self._extract_json(buffer)

            if not data:
                raise ValueError("No valid JSON found in LLM output")

            actions = []

            for item in data.get("actions", []):
                action_type = item.get("type")

                if action_type == "web_search":
                    actions.append(
                        Action(
                            type="web_search",
                            payload={"query": item.get("query")},
                        )
                    )

                elif action_type == "write_memory":
                    actions.append(
                        Action(
                            type="write_memory",
                            payload={"content": item.get("content")},
                        )
                    )

                elif action_type == "respond":
                    actions.append(Action(type="respond"))

                else:
                    logger.warning(
                        "Unknown action type from LLM: %r",
                        action_type,
                    )

            if actions:
                logger.info(
                    "LLMPlanner produced %d actions (timeout=%s)",
                    len(actions),
                    timed_out,
                )
                return Plan(actions=actions)

            logger.warning("LLMPlanner parsed JSON but produced no actions")

        except Exception:
            logger.exception(
                "LLMPlanner failed to parse output as JSON. Raw output: %r",
                buffer,
            )

        logger.info("LLMPlanner fallback to default respond")
        return Plan(actions=[Action(type="respond")])

    # ============================================================
    # Helpers
    # ============================================================

    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract the first JSON object found in the text.
        This makes the planner robust against:
        - streamed partial output
        - extra commentary
        - markdown fences
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _format_perception(self, perception: dict) -> str:
        if not perception:
            return "No additional perception available."

        lines = []
        for key, entry in perception.items():
            try:
                age = f"{entry.age:.1f}s"
                value = entry.value
            except Exception:
                age = "unknown"
                value = entry

            lines.append(f"- {key}: {value} (age: {age})")

        return "\n".join(lines)
