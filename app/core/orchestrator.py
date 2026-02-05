import uuid
import logging
from typing import Dict, Optional

from app.core.events import AssistantSpeechEvent, AssistantStateEvent
from app.core.assistant_state import AssistantState
from app.core.intents import is_memory_command, extract_memory_content

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
        tools: Dict[str, object],
        summary_trigger: int = 10,
    ):
        self.llm = llm
        self.context_builder = context_builder
        self.history = history_store
        self.memory = memory_store
        self.summary_store = summary_store
        self.summarizer = summarizer
        self.planner = planner
        self.tools = tools
        self.summary_trigger = summary_trigger

        self.session_id = str(uuid.uuid4())[:8]
        logger.info("[%s] Orchestrator initialized", self.session_id)

    # ============================================================
    # Public entry point
    # ============================================================

    def handle_user_input(self, user_text: str):
        logger.info("[%s] User input received: %s", self.session_id, user_text)
        yield AssistantStateEvent(state=AssistantState.THINKING)

        # 1. Persist user input
        self.history.add(self.session_id, "user", user_text)

        # 2. Explicit memory command (early exit)
        for result in self._handle_explicit_memory_command(user_text):
            if result:
                return
            return

        # 3. Planning
        decision = self._plan(user_text)

        # 4. Optional tool execution
        tool_context = yield from self._maybe_run_tool(decision, user_text)

        # 5. Context construction
        messages = self._build_context(user_text, tool_context)

        # 6. LLM streaming response
        response = yield from self._stream_response(messages)

        # 7. Persist assistant response
        self.history.add(self.session_id, "assistant", response)
        yield AssistantSpeechEvent(text=response, is_final=True)
        yield AssistantStateEvent(state=AssistantState.IDLE)

        # 8. Post-processing (summarization)
        self._maybe_summarize()

    # ============================================================
    # Planning & tools
    # ============================================================

    def _plan(self, user_text: str):
        logger.info("[%s] Running planner", self.session_id)
        decision = self.planner.decide(user_text)
        logger.info("[%s] Planner decision: %s", self.session_id, decision.action)
        return decision

    def _maybe_run_tool(self, decision, user_text: str) -> Optional[str]:
        tool = self.tools.get(decision.action)

        if not tool:
            return None

        if not tool.is_available:
            logger.info(
                "[%s] Tool '%s' requested but unavailable, skipping",
                self.session_id,
                decision.action,
            )
            return None

        yield AssistantStateEvent(state=AssistantState.SEARCHING)
        logger.info("[%s] Running tool: %s", self.session_id, decision.action)

        try:
            context = tool.run(decision.query or user_text)
            if context:
                logger.debug(
                    "[%s] Tool '%s' returned context (%d chars)",
                    self.session_id,
                    decision.action,
                    len(context),
                )
            return context

        except Exception:
            logger.warning(
                "[%s] Tool '%s' failed during execution, falling back",
                self.session_id,
                decision.action,
            )
            return None

    # ============================================================
    # Context & response
    # ============================================================

    def _build_context(self, user_text: str, tool_context: Optional[str]):
        logger.info("[%s] Building context", self.session_id)
        messages = self.context_builder.build(
            session_id=self.session_id,
            user_text=user_text,
            tool_context=tool_context,
        )
        logger.debug("[%s] Context message count: %d", self.session_id, len(messages))
        return messages

    def _stream_response(self, messages):
        logger.info("[%s] Calling LLM (streaming)", self.session_id)
        yield AssistantStateEvent(state=AssistantState.RESPONDING)

        buffer = ""
        for chunk in self.llm.stream_chat(messages):
            buffer += chunk
            yield AssistantSpeechEvent(text=chunk)

        logger.info("[%s] LLM response complete (%d chars)", self.session_id, len(buffer))
        return buffer

    # ============================================================
    # Memory & summarization
    # ============================================================

    def _handle_explicit_memory_command(self, user_text: str) -> bool:
        if not is_memory_command(user_text):
            return False

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
        yield AssistantSpeechEvent(text=response)
        yield AssistantSpeechEvent(text=response, is_final=True)
        return True

    def _maybe_summarize(self):
        logger.debug("[%s] Checking summarization conditions", self.session_id)

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
