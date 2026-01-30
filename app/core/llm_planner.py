import json
import time
from app.core.planner import PlannerDecision


class LLMPlanner:
    def __init__(self, llm, timeout_ms: int = 1500):
        self.llm = llm
        self.timeout_ms = timeout_ms

    def decide(self, user_text: str) -> PlannerDecision | None:
        start = time.time()

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a planner for an AI assistant.\n"
                    "Decide whether web search is required.\n"
                    "Output ONLY valid JSON.\n"
                    "{ \"action\": \"respond | web_search\", "
                    "\"query\": string | null, "
                    "\"reason\": string }"
                )
            },
            {"role": "user", "content": user_text}
        ]

        buffer = ""
        for chunk in self.llm.stream_chat(prompt):
            buffer += chunk
            if (time.time() - start) * 1000 > self.timeout_ms:
                return None  # timeout → fallback

        try:
            data = json.loads(buffer.strip())
            return PlannerDecision(
                action=data["action"],
                query=data.get("query"),
                reason=data.get("reason"),
            )
        except Exception:
            return None
