import uuid
import logging
import time
from typing import Generator, Optional, Dict

from app.core.events import AssistantSpeechEvent, AssistantStateEvent
from app.core.assistant_state import AssistantState
from app.core.intents import is_memory_command, extract_memory_content
from app.core.actions import Action
from app.core.plan import Plan
from app.perception.state import PerceptionState

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

        self.perception = PerceptionState()  # NEW

        self.session_id = str(uuid.uuid4())[:8]

        logger.info(
            "[%s] Orchestrator initialized (tools=%d, summary_trigger=%d)",
            self.session_id,
            len(tools),
            summary_trigger,
        )

    # ============================================================
    # Public entry point
    # ============================================================

    def handle_user_input(self, user_text: str):
        start_ts = time.perf_counter()

        logger.info(
            "[%s] User input received (len=%d)",
            self.session_id,
            len(user_text),
        )
        logger.debug("[%s] User input text: %r", self.session_id, user_text)

        yield AssistantStateEvent(state=AssistantState.THINKING)

        # --------------------------------------------------------
        # 1. Update perception (NEW)
        # --------------------------------------------------------
        self.perception.update(
            "user.input",
            {
                "text": user_text,
                "source": "keyboard",  # later: voice
            },
        )

        # --------------------------------------------------------
        # 2. Persist user input
        # --------------------------------------------------------
        self.history.add(self.session_id, "user", user_text)
        logger.debug("[%s] User input persisted to history", self.session_id)

        # --------------------------------------------------------
        # 3. Planning (decide actions)
        # --------------------------------------------------------
        perception_snapshot = self.perception.snapshot()  # NEW
        plan = self._plan(user_text, perception_snapshot)  # NEW

        logger.debug(
            "[%s] Plan actions: %s",
            self.session_id,
            [action.type for action in plan.actions],
        )

        tool_context: Optional[str] = None

        # --------------------------------------------------------
        # 4. Execute actions
        # --------------------------------------------------------
        for action in plan.actions:
            logger.info(
                "[%s] Executing action '%s'",
                self.session_id,
                action.type,
            )

            if action.type == "web_search":
                tool_context = yield from self._run_tool_action(action, user_text)

            elif action.type == "write_memory":
                self._run_memory_action(action)

            elif action.type == "respond":
                logger.debug(
                    "[%s] Respond action reached, stopping action loop",
                    self.session_id,
                )
                break

            else:
                logger.warning(
                    "[%s] Unknown action '%s', skipping",
                    self.session_id,
                    action.type,
                )

        # --------------------------------------------------------
        # 5. Context construction
        # --------------------------------------------------------
        messages = self._build_context(user_text, tool_context)

        # --------------------------------------------------------
        # 6. LLM streaming response
        # --------------------------------------------------------
        response = yield from self._stream_response(messages)

        # --------------------------------------------------------
        # 7. Persist assistant response
        # --------------------------------------------------------
        self.history.add(self.session_id, "assistant", response)
        logger.debug("[%s] Assistant response persisted to history", self.session_id)

        yield AssistantSpeechEvent(text=response, is_final=True)
        yield AssistantStateEvent(state=AssistantState.IDLE)

        # --------------------------------------------------------
        # 8. Post-processing (summarization)
        # --------------------------------------------------------
        self._maybe_summarize()

        logger.info(
            "[%s] Turn completed (duration=%.2f ms)",
            self.session_id,
            (time.perf_counter() - start_ts) * 1000,
        )

    # ============================================================
    # Planning
    # ============================================================

    def _plan(self, user_text: str, perception: dict) -> Plan:  # NEW
        logger.info("[%s] Running planner", self.session_id)

        try:
            plan = self.planner.decide(
                user_text=user_text,
                perception=perception,  # NEW
            )
        except Exception:
            logger.exception("[%s] Planner failed", self.session_id)
            raise

        logger.info(
            "[%s] Planner produced %d actions",
            self.session_id,
            len(plan.actions),
        )
        return plan

    # ============================================================
    # Action execution
    # ============================================================

    def _run_tool_action(
        self,
        action: Action,
        user_text: str,
    ) -> Generator[AssistantStateEvent, None, Optional[str]]:

        tool = self.tools.get(action.type)

        if not tool:
            logger.warning(
                "[%s] Tool '%s' not registered, skipping",
                self.session_id,
                action.type,
            )
            return None

        if not tool.is_available:
            logger.warning(
                "[%s] Tool '%s' unavailable, skipping",
                self.session_id,
                action.type,
            )
            return None

        yield AssistantStateEvent(state=AssistantState.SEARCHING)

        logger.info("[%s] Running tool '%s'", self.session_id, action.type)

        start_ts = time.perf_counter()

        try:
            query = (action.payload or {}).get("query") or user_text
            logger.debug("[%s] Tool '%s' query: %r", self.session_id, action.type, query)

            context = tool.run(query)

            logger.info(
                "[%s] Tool '%s' completed (duration=%.2f ms)",
                self.session_id,
                action.type,
                (time.perf_counter() - start_ts) * 1000,
            )

            if context:
                logger.debug(
                    "[%s] Tool '%s' returned context (%d chars)",
                    self.session_id,
                    action.type,
                    len(context),
                )
            else:
                logger.debug("[%s] Tool '%s' returned no context", self.session_id, action.type)

            return context

        except Exception:
            logger.exception(
                "[%s] Tool '%s' failed, falling back",
                self.session_id,
                action.type,
            )
            return None

    def _run_memory_action(self, action: Action):
        logger.debug("[%s] Processing memory action", self.session_id)

        decision = self.memory_policy.decide_from_action(action.payload or {})

        if not decision:
            logger.debug("[%s] Memory action ignored by policy", self.session_id)
            return

        self.memory.add(
            content=decision.content,
            category=decision.category,
            importance=decision.importance,
        )

        logger.info(
            "[%s] Memory written (category=%s, importance=%d)",
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

        logger.debug(
            "[%s] Context built (messages=%d, tool_context=%s)",
            self.session_id,
            len(messages),
            bool(tool_context),
        )
        return messages

    def _stream_response(self, messages):
        logger.info("[%s] Calling LLM (streaming)", self.session_id)
        yield AssistantStateEvent(state=AssistantState.RESPONDING)

        buffer = ""
        start_ts = time.perf_counter()

        for chunk in self.llm.stream_chat(messages):
            buffer += chunk
            yield AssistantSpeechEvent(text=chunk)

        logger.info(
            "[%s] LLM response complete (chars=%d, duration=%.2f ms)",
            self.session_id,
            len(buffer),
            (time.perf_counter() - start_ts) * 1000,
        )
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

        try:
            summary = self.summarizer.summarize(summary_input)
        except Exception:
            logger.exception("[%s] Summarization failed", self.session_id)
            return

        self.summary_store.set(self.session_id, summary)

        logger.info(
            "[%s] History summarized (%d chars)",
            self.session_id,
            len(summary),
        )
