import json
from app.core.planner import PlannerDecision


class LLMPlanner:
    def __init__(self, llm):
        self.llm = llm

    def decide(self, user_text: str) -> PlannerDecision | None:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a planner for an AI assistant.\n\n"
                    "Rules:\n"
                    "- You must choose exactly one action: \"respond\" or \"web_search\".\n"
                    "- Choose \"web_search\" ONLY if up-to-date or factual external information is required.\n"
                    "- Choose \"respond\" for opinions, explanations, formatting questions, or casual conversation.\n"
                    "- Do NOT answer the user.\n"
                    "- Do NOT include markdown.\n"
                    "- Output ONLY valid JSON.\n\n"
                    "JSON schema:\n"
                    "{\n"
                    "  \"action\": \"respond | web_search\",\n"
                    "  \"query\": string | null,\n"
                    "  \"reason\": string\n"
                    "}"
                )
            },
            {
                "role": "user",
                "content": user_text
            }
        ]

        buffer = ""
        for chunk in self.llm.stream_chat(prompt):
            buffer += chunk

        try:
            data = json.loads(buffer.strip())
            return PlannerDecision(
                action=data["action"],
                query=data.get("query"),
                reason=data.get("reason"),
            )
        except Exception:
            return None
