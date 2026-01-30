import uuid
from app.core.events import AssistantSpeechEvent
from app.core.intents import is_memory_command, extract_memory_content
from app.tools.search_formatter import format_search_results


class Orchestrator:
    def __init__(
        self,
        llm,
        context_builder,
        history_store,
        memory_store,
        summary_store,
        summarizer,
        planner,
        web_search,
        search_summarizer,
        summary_trigger: int = 10,
    ):
        self.llm = llm
        self.context_builder = context_builder
        self.history = history_store
        self.memory = memory_store
        self.summary_store = summary_store
        self.summarizer = summarizer
        self.search_summarizer = search_summarizer


        self.planner = planner
        self.web_search = web_search
        self.summary_trigger = summary_trigger
        self.session_id = str(uuid.uuid4())

    def handle_user_input(self, user_text: str):
        # 1. Always store user input
        self.history.add(self.session_id, "user", user_text)

        # 2. Manual memory write handling (early exit)
        if is_memory_command(user_text):
            memory_content = extract_memory_content(user_text)

            if memory_content:
                self.memory.add(content=memory_content, importance=2)
                response = "Got it. I’ll remember that."
            else:
                response = "What would you like me to remember?"

            self.history.add(self.session_id, "assistant", response)
            yield AssistantSpeechEvent(text=response)
            yield AssistantSpeechEvent(text=response, is_final=True)
            return

        # 3. PLANNING STEP (NEW)
        decision = self.planner.decide(user_text)
        web_context = None

        if decision.use_web:
            results = self.web_search.search(decision.query)
            summary = self.search_summarizer.summarize(results)

        # 4. Build context (ONLY place prompt is assembled)
        messages = self.context_builder.build(
            session_id=self.session_id,
            user_text=user_text,
            web_context=web_context,
        )

        # 5. Stream LLM response
        buffer = ""
        for chunk in self.llm.stream_chat(messages):
            buffer += chunk
            yield AssistantSpeechEvent(text=chunk)

        # 6. Store assistant response
        self.history.add(self.session_id, "assistant", buffer)
        yield AssistantSpeechEvent(text=buffer, is_final=True)

        # 7. Maybe summarize history
        self._maybe_summarize()

    def _maybe_summarize(self):
        if self.summary_store.get(self.session_id):
            return

        history = self.history.get_recent(
            session_id=self.session_id,
            limit=100,
        )

        if len(history) < self.summary_trigger:
            return

        summary_input = [
            {"role": row["role"], "content": row["content"]}
            for row in history
        ]

        summary = self.summarizer.summarize(summary_input)
        self.summary_store.upsert(self.session_id, summary)
