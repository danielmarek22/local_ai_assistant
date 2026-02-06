import json
import time
import logging

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

    def decide(self, user_text: str) -> Plan:
        start_ts = time.perf_counter()

        logger.info(
            "LLMPlanner invoked (len=%d)",
            len(user_text),
        )
        logger.debug("LLMPlanner input text: %r", user_text)

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a planner for an AI assistant.\n"
                    "Decide what actions to take.\n"
                    "Output ONLY valid JSON.\n\n"
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

        logger.debug("Starting LLM stream for planning")

        for chunk in self.llm.stream_chat(prompt):
            buffer += chunk

            elapsed_ms = (time.perf_counter() - start_ts) * 1000
            if elapsed_ms > self.timeout_ms:
                timed_out = True
                logger.warning(
                    "LLMPlanner timeout after %.2f ms",
                    elapsed_ms,
                )
                break

        logger.debug(
            "LLMPlanner raw output (%d chars): %r",
            len(buffer),
            buffer,
        )

        try:
            data = json.loads(buffer.strip())
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
                    "LLMPlanner produced %d actions (timeout=%s, duration=%.2f ms)",
                    len(actions),
                    timed_out,
                    (time.perf_counter() - start_ts) * 1000,
                )
                return Plan(actions=actions)

            logger.warning(
                "LLMPlanner parsed JSON but produced no valid actions",
            )

        except Exception:
            logger.exception(
                "LLMPlanner failed to parse output as JSON",
            )

        # Fallback: always respond
        logger.info(
            "LLMPlanner fallback to default respond (duration=%.2f ms)",
            (time.perf_counter() - start_ts) * 1000,
        )

        return Plan(actions=[Action(type="respond")])
