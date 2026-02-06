import uuid
import logging
from typing import Dict, Optional

from app.core.events import AssistantSpeechEvent, AssistantStateEvent
from app.core.assistant_state import AssistantState
from app.core.intents import is_memory_command, extract_memory_content
from app.core.actions import Action
from app.core.plan import Plan

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
        memory_policy,
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
        self.memory_policy = memory_policy

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

        # 2. Planning (decide actions)
        plan = self._plan(user_text)

        logger.debug(
            "[%s] Plan actions: %s",
            self.session_id,
            [action.type for action in plan.actions],
        )

        tool_context: Optional[str] = None

        # 3. Execute actions in order
        for action in plan.actions:
            if action.type == "web_search":
                tool_context = yield from self._run_tool_action(action, user_text)

            elif action.type == "write_memory":
                self._run_memory_action(action)

            elif action.type == "respond":
                break

            else:
                logger.warning(
                    "[%s] Unknown action '%s', skipping",
                    self.session_id,
                    action.type,
                )

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
    # Planning
    # ============================================================

    def _plan(self, user_text: str) -> Plan:
        logger.info("[%s] Running planner", self.session_id)
        plan = self.planner.decide(user_text)
        logger.info(
            "[%s] Planner produced %d actions",
            self.session_id,
            len(plan.actions),
        )
        return plan

    # ============================================================
    # Action execution
    # ============================================================

    def _run_tool_action(self, action: Action, user_text: str) -> Optional[str]:
        tool = self.tools.get(action.type)

        if not tool:
            logger.info(
                "[%s] Tool '%s' not registered, skipping",
                self.session_id,
                action.type,
            )
            return None

        if not tool.is_available:
            logger.info(
                "[%s] Tool '%s' unavailable, skipping",
                self.session_id,
                action.type,
            )
            return None

        yield AssistantStateEvent(state=AssistantState.SEARCHING)
        logger.info("[%s] Running tool: %s", self.session_id, action.type)

        try:
            query = (action.payload or {}).get("query") or user_text
            context = tool.run(query)

            if context:
                logger.debug(
                    "[%s] Tool '%s' returned context (%d chars)",
                    self.session_id,
                    action.type,
                    len(context),
                )

            return context

        except Exception:
            logger.warning(
                "[%s] Tool '%s' failed, falling back",
                self.session_id,
                action.type,
            )
            return None

    def _run_memory_action(self, action: Action):
        decision = self.memory_policy.decide_from_action(action.payload or {})

        if not decision:
            logger.debug("[%s] Memory action ignored (no decision)", self.session_id)
            return

        self.memory.add(
            content=decision.content,
            category=decision.category,
            importance=decision.importance,
        )

        logger.info(
            "[%s] Memory written via policy (category=%s, importance=%d)",
            self.session_id,
            decision.category,
            decision.importance,
    )


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
        print(messages)
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
    # Summarization
    # ============================================================

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
