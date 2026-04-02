import json
import time
import logging
import concurrent.futures
from typing import Optional, Literal

from pydantic import BaseModel, ValidationError, model_validator

from app.core.actions import Action, ActionType
from app.core.plan import Plan
from app.logging import trace_event

logger = logging.getLogger("llm_planner")


class PlannerActionSpec(BaseModel):
    type: ActionType
    query: Optional[str] = None
    content: Optional[str] = None

    class Config:
        extra = "forbid"

    @model_validator(mode="after")
    def validate_fields_for_type(self):
        if self.type == ActionType.WEB_SEARCH:
            if not self.query:
                raise ValueError("web_search requires 'query'")
            if self.content is not None:
                raise ValueError("web_search forbids 'content'")
            
        elif self.type == ActionType.WRITE_MEMORY:
            if not self.content:
                raise ValueError("write_memory requires 'content'")
            if self.query is not None:
                raise ValueError("write_memory forbids 'query'")
            
        elif self.type == ActionType.RESPOND:
            if self.query is not None or self.content is not None:
                raise ValueError("respond forbids 'query' and 'content'")

        return self


class PlannerOutput(BaseModel):
    actions: list[PlannerActionSpec]

    class Config:
        extra = "forbid"


class LLMPlanner:
    def __init__(self, llm, timeout_ms: int = 6000):
        self.llm = llm
        self.timeout_ms = timeout_ms
        logger.info("LLMPlanner initialized (timeout_ms=%d)", timeout_ms)

    def decide(self, user_text: str, perception: dict) -> Plan:
        start_ts = time.perf_counter()

        logger.info("LLMPlanner invoked (len=%d)", len(user_text))

        # VRAM Saver: Truncate user text if it's unreasonably long
        safe_user_text = user_text[:1000] 
        perception_text = self._format_perception(perception)

        # Minified, highly-instructive prompt for smart memory extraction
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an internal system router. Your ONLY job is to output a strictly formatted JSON object to trigger the correct system action. Do NOT act like a conversational assistant.\n\n"
                    "Context (Review this to avoid duplicate actions):\n"
                    f"{perception_text}\n\n"
                    "Allowed Actions:\n"
                    "1. web_search: Use to find real-time facts/news. (Requires a 'query' string).\n"
                    "2. write_memory: Use ONLY to save enduring, long-term facts about the user. (Requires a 'content' string).\n"
                    "3. respond: Use when you just need to talk to the user.\n\n"
                    "CRITICAL RULES:\n"
                    "- Output ONLY valid JSON matching the examples perfectly.\n"
                    "- NEVER include a 'response', 'text', or 'message' field in your JSON.\n"
                    "- Do NOT draft the actual reply to the user. The response generation happens in a later step.\n\n"
                    "Example 1:\n"
                    '{"actions": [{"type": "web_search", "query": "weather in Tokyo"}]}\n\n'
                    "Example 2:\n"
                    '{"actions": [{"type": "write_memory", "content": "User prefers Python."}]}\n\n'
                    "Example 3 (Strictly use this format for responding):\n"
                    '{"actions": [{"type": "respond"}]}'
                ),
            },
            {"role": "user", "content": safe_user_text},
        ]
        trace_event(
            "llm_planner",
            "planner_call",
            payload={
                "user_text": user_text,
                "safe_user_text": safe_user_text,
                "perception": perception,
                "prompt": prompt,
            },
        )
        buffer = self._call_llm_with_timeout(prompt)

        if buffer is None:
            logger.warning("LLMPlanner timed out after %.2fs", self.timeout_ms / 1000)
            trace_event("llm_planner", "planner_timeout", payload={"timeout_ms": self.timeout_ms})
            return self._fallback_plan()

        try:
            data = self._extract_json(buffer)
            if not data:
                raise ValueError("No JSON found")

            parsed = self._validate_output(data)
            actions = []

            for item in parsed.actions:
                if item.type == ActionType.WEB_SEARCH:
                    actions.append(Action(type=ActionType.WEB_SEARCH, payload={"query": item.query}))
                elif item.type == ActionType.WRITE_MEMORY:
                    actions.append(Action(type=ActionType.WRITE_MEMORY, payload={"content": item.content}))
                elif item.type == ActionType.RESPOND:
                    actions.append(Action(type=ActionType.RESPOND))

            if actions:
                logger.info(
                    "LLMPlanner produced %d actions (%.2f ms)",
                    len(actions),
                    (time.perf_counter() - start_ts) * 1000,
                )
                trace_event(
                    "llm_planner",
                    "planner_result",
                    payload={
                        "raw_output": buffer,
                        "extracted_json": data,
                        "actions": [
                            {"type": action.type.value, "payload": action.payload}
                            for action in actions
                        ],
                    },
                )
                return Plan(actions=actions)

        except Exception as e:
            logger.warning("LLMPlanner parsing failed: %s. Raw: %r", e, buffer)
            trace_event(
                "llm_planner",
                "planner_parse_failure",
                payload={"error": str(e), "raw_output": buffer},
            )

        return self._fallback_plan()

    # ============================================================
    # Helpers
    # ============================================================

    def _fallback_plan(self) -> Plan:
        return Plan(actions=[Action(type=ActionType.RESPOND)])

    def _call_llm_with_timeout(self, prompt: list[dict]) -> Optional[str]:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        timeout_seconds = self.timeout_ms / 1000.0
        
        future = executor.submit(
            self.llm.chat,
            prompt,
            think_override=False,
            options_override={"temperature": 0.0, "num_predict": 150},
            timeout_override=timeout_seconds,
            max_retries_override=0,
        )
        
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _validate_output(self, data: dict) -> PlannerOutput:
        try:
            return PlannerOutput.model_validate(data)
        except AttributeError:
            return PlannerOutput.parse_obj(data)
        except ValidationError:
            raise

    def _extract_json(self, text: str) -> Optional[dict]:
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

    def _format_perception(self, perception: dict, max_items: int = 5) -> str:
        if not perception:
            return "None"

        lines = []
        for i, (key, entry) in enumerate(perception.items()):
            if i >= max_items:
                lines.append("- [Truncated to save context]")
                break
            try:
                value = str(entry)[:150] # Increased slightly to allow memories to fit
            except Exception:
                value = str(entry)[:150]

            lines.append(f"- {key}: {value}")

        return "\n".join(lines)
