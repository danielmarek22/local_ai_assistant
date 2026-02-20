import json
import time
import logging
import concurrent.futures
from typing import Optional, Literal

from pydantic import BaseModel, ValidationError

from app.core.actions import Action
from app.core.plan import Plan

logger = logging.getLogger("llm_planner")


class PlannerActionSpec(BaseModel):
    type: Literal["web_search", "write_memory", "respond"]
    query: Optional[str] = None
    content: Optional[str] = None


class PlannerOutput(BaseModel):
    actions: list[PlannerActionSpec]


class LLMPlanner:
    def __init__(self, llm, timeout_ms: int = 4000):
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

        buffer = self._call_llm_with_timeout(prompt)

        if buffer is None:
            logger.warning(
                "LLMPlanner timed out after %.2f seconds",
                self.timeout_ms / 1000,
            )
            return self._fallback_plan()

        logger.debug("LLMPlanner raw output: %r", buffer)

        try:
            data = self._extract_json(buffer)

            if not data:
                raise ValueError("No valid JSON found in LLM output")

            parsed = self._validate_output(data)
            actions = []

            for item in parsed.actions:
                action_type = item.type

                if action_type == "web_search":
                    if not item.query:
                        logger.warning("Skipping web_search action with empty query")
                        continue
                    actions.append(
                        Action(
                            type="web_search",
                            payload={"query": item.query},
                        )
                    )

                elif action_type == "write_memory":
                    if not item.content:
                        logger.warning("Skipping write_memory action with empty content")
                        continue
                    actions.append(
                        Action(
                            type="write_memory",
                            payload={"content": item.content},
                        )
                    )

                elif action_type == "respond":
                    actions.append(Action(type="respond"))

            if actions:
                logger.info(
                    "LLMPlanner produced %d actions (%.2f ms)",
                    len(actions),
                    (time.perf_counter() - start_ts) * 1000,
                )
                return Plan(actions=actions)

            logger.warning("LLMPlanner parsed JSON but produced no actions")

        except Exception:
            logger.exception(
                "LLMPlanner failed to parse output as JSON. Raw output: %r",
                buffer,
            )

        return self._fallback_plan()

    # ============================================================
    # Helpers
    # ============================================================

    def _fallback_plan(self) -> Plan:
        logger.info("LLMPlanner fallback to default respond")
        return Plan(actions=[Action(type="respond")])

    def _call_llm_with_timeout(self, prompt: list[dict]) -> Optional[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.llm.chat, prompt)
            try:
                return future.result(timeout=self.timeout_ms / 1000)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return None

    def _validate_output(self, data: dict) -> PlannerOutput:
        try:
            return PlannerOutput.model_validate(data)
        except AttributeError:
            return PlannerOutput.parse_obj(data)
        except ValidationError:
            logger.exception("LLMPlanner output failed schema validation")
            raise

    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract the first valid JSON object from arbitrary text output.
        """
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
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
