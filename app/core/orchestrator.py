import uuid
import logging
from app.core.events import AssistantSpeechEvent
from app.core.intents import is_memory_command, extract_memory_content
from app.core.assistant_state import AssistantState
from app.core.events import AssistantStateEvent

logger = logging.getLogger("orchestrator")


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

        self.session_id = str(uuid.uuid4())[:8]
        logger.info("[%s] Orchestrator initialized", self.session_id)

    def handle_user_input(self, user_text: str):
        logger.info("[%s] User input received: %s", self.session_id, user_text)
        yield AssistantStateEvent(state=AssistantState.THINKING)


        # ------------------------------------------------------------------
        # 1. Persist user input
        # ------------------------------------------------------------------
        self.history.add(self.session_id, "user", user_text)

        # ------------------------------------------------------------------
        # 2. Manual memory write (explicit, early exit)
        # ------------------------------------------------------------------
        if is_memory_command(user_text):
            logger.info("[%s] Memory command detected", self.session_id)

            memory_content = extract_memory_content(user_text)

            if memory_content:
                self.memory.add(content=memory_content, importance=2)
                response = "Got it. I’ll remember that."
                logger.info("[%s] Memory stored: %s", self.session_id, memory_content)
            else:
                response = "What would you like me to remember?"
                logger.info("[%s] Memory command without content", self.session_id)

            self.history.add(self.session_id, "assistant", response)

            # Double-yield is intentional (stream + final)
            yield AssistantSpeechEvent(text=response)
            yield AssistantSpeechEvent(text=response, is_final=True)
            return

        # ------------------------------------------------------------------
        # 3. Planning phase
        # ------------------------------------------------------------------
        logger.info("[%s] Running planner", self.session_id)
        decision = self.planner.decide(user_text)
        logger.info("[%s] Planner decision: %s", self.session_id, decision.action)

        web_context = None

        if decision.action == "web_search":
            yield AssistantStateEvent(state=AssistantState.SEARCHING)
            logger.info("[%s] Performing web search", self.session_id)

            try:
                results = self.web_search.search(decision.query or user_text)
                summary = self.search_summarizer.summarize(results)
                web_context = f"External information:\n{summary}"

                logger.debug(
                    "[%s] Web context injected (%d chars)",
                    self.session_id,
                    len(web_context),
                )

            except Exception as e:
                logger.warning(
                    "[%s] Web search failed, falling back to local response: %s",
                    self.session_id,
                    str(e),
                )
                web_context = None

        # ------------------------------------------------------------------
        # 4. Context construction
        # ------------------------------------------------------------------
        logger.info("[%s] Building context", self.session_id)

        messages = self.context_builder.build(
            session_id=self.session_id,
            user_text=user_text,
            web_context=web_context,
        )

        logger.debug("[%s] Context message count: %d", self.session_id, len(messages))

        # ------------------------------------------------------------------
        # 5. LLM streaming response
        # ------------------------------------------------------------------
        logger.info("[%s] Calling LLM (streaming)", self.session_id)
        yield AssistantStateEvent(state=AssistantState.RESPONDING)
        buffer = ""
        
        for chunk in self.llm.stream_chat(messages):
            buffer += chunk
            yield AssistantSpeechEvent(text=chunk)

        logger.info("[%s] LLM response complete (%d chars)", self.session_id, len(buffer))

        # ------------------------------------------------------------------
        # 6. Persist assistant response
        # ------------------------------------------------------------------
        self.history.add(self.session_id, "assistant", buffer)

        yield AssistantSpeechEvent(text=buffer, is_final=True)
        yield AssistantStateEvent(state=AssistantState.IDLE)

        # ------------------------------------------------------------------
        # 7. History summarization (if needed)
        # ------------------------------------------------------------------
        self._maybe_summarize()

    def _maybe_summarize(self):
        logger.debug("[%s] Checking summarization conditions", self.session_id)

        # Do not summarize twice
        if self.summary_store.get(self.session_id):
            logger.debug("[%s] Summary already exists, skipping", self.session_id)
            return

        history = self.history.get_recent(
            session_id=self.session_id,
            limit=100,
        )

        if len(history) < self.summary_trigger:
            logger.debug(
                "[%s] History length (%d) below trigger (%d)",
                self.session_id,
                len(history),
                self.summary_trigger,
            )
            return

        logger.info("[%s] Summarizing conversation history", self.session_id)

        summary_input = [
            {"role": row["role"], "content": row["content"]}
            for row in history
        ]

        summary = self.summarizer.summarize(summary_input)
        self.summary_store.set(self.session_id, summary)

        logger.info("[%s] History summarized (%d chars)", self.session_id, len(summary))
