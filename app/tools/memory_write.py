import logging

from app.core.actions import Action, ActionType
from app.services.memory_action_handler import MemoryActionHandler

logger = logging.getLogger("memory_write_tool")


class MemoryWriteTool:
    """
    Native Ollama tool for persisting an enduring fact into semantic memory.
    Accepts a structured payload and reuses the existing memory policy/handler path.
    """

    name = "write_memory"
    description = "Persist an enduring fact or preference into long-term semantic memory."
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact or instruction to persist in memory.",
            },
            "category": {
                "type": "string",
                "description": "Optional memory category, such as general or preference.",
            },
            "importance": {
                "type": "integer",
                "description": "Optional importance score used for ranking and freshness.",
            },
        },
        "required": ["content"],
    }

    def __init__(self, memory_action_handler: MemoryActionHandler):
        self.memory_action_handler = memory_action_handler

    @property
    def is_available(self) -> bool:
        return True

    def run(self, payload: dict | str) -> str | None:
        if isinstance(payload, str):
            payload = {"content": payload}

        if not isinstance(payload, dict):
            logger.warning("write_memory tool received non-dict payload: %r", payload)
            return "Memory write failed: invalid payload."

        action = Action(type=ActionType.WRITE_MEMORY, payload=payload)
        self.memory_action_handler.handle(session_id="native_tool", action=action)

        logger.info("Memory write tool stored content: %s", payload.get("content"))
        return f"Memory write accepted: {payload.get('content', '')}"
