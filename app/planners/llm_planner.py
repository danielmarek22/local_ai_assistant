import json
import time
import logging
import concurrent.futures
from typing import Optional, Literal

from pydantic import BaseModel, ValidationError, model_validator

from app.core.actions import Action
from app.core.plan import Plan

logger = logging.getLogger("llm_planner")


class PlannerActionSpec(BaseModel):
    type: Literal["web_search", "write_memory", "respond"]
    query: Optional[str] = None
    content: Optional[str] = None

    class Config:
        extra = "forbid"

    @model_validator(mode="after")
    def validate_fields_for_type(self):
        if self.type == "web_search":
            if not self.query:
                raise ValueError("web_search requires 'query'")
            if self.content is not None:
                raise ValueError("web_search forbids 'content'")
            
        elif self.type == "write_memory":
            if not self.content:
                raise ValueError("write_memory requires 'content'")
            if self.query is not None:
                raise ValueError("write_memory forbids 'query'")
            
        elif self.type == "respond":
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
                    "You are an AI planner. Decide the next action.\n"
                    "Output ONLY valid JSON.\n\n"
                    "Context (Check this before searching or saving!):\n"
                    f"{perception_text}\n\n"
                    "Allowed Actions:\n"
                    "- web_search: Use to find real-time facts/news. (requires 'query')\n"
                    "- write_memory: Use ONLY to save enduring, long-term facts about the user (e.g., name, tech stack, preferences). Rephrase into a concise rule. Do NOT save temporary states, and do NOT save facts already present in the Context. (requires 'content')\n"
                    "- respond: Use to just talk to the user.\n\n"
                    "Example 1:\n"
                    '{"actions": [{"type": "web_search", "query": "weather in Tokyo"}]}\n'
                    "Example 2:\n"
                    '{"actions": [{"type": "write_memory", "content": "User prefers concise Python code without markdown."}]}\n'
                    "Example 3:\n"
                    '{"actions": [{"type": "respond"}]}'
                ),
            },
            {"role": "user", "content": safe_user_text},
        ]

        buffer = self._call_llm_with_timeout(prompt)

        if buffer is None:
            logger.warning("LLMPlanner timed out after %.2fs", self.timeout_ms / 1000)
            return self._fallback_plan()

        try:
            data = self._extract_json(buffer)
            if not data:
                raise ValueError("No JSON found")

            parsed = self._validate_output(data)
            actions = []

            for item in parsed.actions:
                if item.type == "web_search":
                    actions.append(Action(type="web_search", payload={"query": item.query}))
                elif item.type == "write_memory":
                    actions.append(Action(type="write_memory", payload={"content": item.content}))
                elif item.type == "respond":
                    actions.append(Action(type="respond"))

            if actions:
                logger.info(
                    "LLMPlanner produced %d actions (%.2f ms)",
                    len(actions),
                    (time.perf_counter() - start_ts) * 1000,
                )
                return Plan(actions=actions)

        except Exception as e:
            logger.warning("LLMPlanner parsing failed: %s. Raw: %r", e, buffer)

        return self._fallback_plan()

    # ============================================================
    # Helpers
    # ============================================================

    def _fallback_plan(self) -> Plan:
        return Plan(actions=[Action(type="respond")])

    def _call_llm_with_timeout(self, prompt: list[dict]) -> Optional[str]:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Pass the strict routing parameters inside an options dictionary
        future = executor.submit(
            self.llm.chat, 
            prompt, 
            False,  # think_override=False (we don't want the planner wasting tokens thinking)
            {"temperature": 0.0, "num_predict": 150}  # options_override
        )
        
        try:
            return future.result(timeout=self.timeout_ms / 1000)
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
                value = str(entry.value)[:150] # Increased slightly to allow memories to fit
            except Exception:
                value = str(entry)[:150]

            lines.append(f"- {key}: {value}")

        return "\n".join(lines)