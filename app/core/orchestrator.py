from llm.base import LLMClient
from core.events import AssistantSpeechEvent
import uuid

class Orchestrator:
    def __init__(self, llm, system_prompt: str, history_store, memory_store):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history = history_store
        self.memory = memory_store
        self.session_id = str(uuid.uuid4())

    def handle_user_input(self, user_text: str):
        # Store user message
        self.history.add(self.session_id, "user", user_text)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]

        buffer = ""

        for chunk in self.llm.stream_chat(messages):
            buffer += chunk
            yield AssistantSpeechEvent(text=chunk)

        # Store assistant response
        self.history.add(self.session_id, "assistant", buffer)

        yield AssistantSpeechEvent(text=buffer, is_final=True)
