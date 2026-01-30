from llm.base import LLMClient
from core.events import AssistantSpeechEvent
from core.intents import is_memory_command, extract_memory_content
import uuid

class Orchestrator:
    def __init__(self, llm, system_prompt: str, history_store, memory_store):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history = history_store
        self.memory = memory_store
        self.session_id = str(uuid.uuid4())

    def handle_user_input(self, user_text: str):
            # Always store user input
            self.history.add(self.session_id, "user", user_text)

            # ---- MEMORY COMMAND HANDLING ----
            if is_memory_command(user_text):
                memory_content = extract_memory_content(user_text)

                if memory_content:
                    self.memory.add(
                        content=memory_content,
                        importance=2
                    )

                    response = "Got it. I’ll remember that."

                    self.history.add(self.session_id, "assistant", response)
                    yield AssistantSpeechEvent(text=response)
                    yield AssistantSpeechEvent(text=response, is_final=True)
                    return

                else:
                    response = "What would you like me to remember?"
                    self.history.add(self.session_id, "assistant", response)
                    yield AssistantSpeechEvent(text=response, is_final=True)
                    return
            # --------------------------------

            # Normal LLM flow
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ]

            buffer = ""

            for chunk in self.llm.stream_chat(messages):
                buffer += chunk
                yield AssistantSpeechEvent(text=chunk)

            self.history.add(self.session_id, "assistant", buffer)
            yield AssistantSpeechEvent(text=buffer, is_final=True)
