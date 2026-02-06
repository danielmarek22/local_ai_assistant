import json
import time
from app.core.actions import Action
from app.core.plan import Plan


class LLMPlanner:
    def __init__(self, llm, timeout_ms: int = 1500):
        self.llm = llm
        self.timeout_ms = timeout_ms

    def decide(self, user_text: str) -> Plan:
        start = time.time()

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
        for chunk in self.llm.stream_chat(prompt):
            buffer += chunk
            if (time.time() - start) * 1000 > self.timeout_ms:
                break  # timeout → fallback

        try:
            data = json.loads(buffer.strip())
            actions = []

            for item in data.get("actions", []):
                if item["type"] == "web_search":
                    actions.append(
                        Action(
                            type="web_search",
                            payload={"query": item.get("query")},
                        )
                    )
                elif item["type"] == "write_memory":
                    actions.append(
                        Action(
                            type="write_memory",
                            payload={"content": item.get("content")},
                        )
                    )
                elif item["type"] == "respond":
                    actions.append(Action(type="respond"))

            if actions:
                return Plan(actions=actions)

        except Exception:
            pass

        # Fallback: always respond
        return Plan(actions=[Action(type="respond")])
