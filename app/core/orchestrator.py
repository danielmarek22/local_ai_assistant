import uuid
import logging
import re
import time
from typing import Generator, Optional, Dict

from app.core.events import (
    AssistantSpeechEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
)
from app.core.assistant_state import AssistantState
from app.core.actions import Action
from app.core.plan import Plan
from app.perception.state import PerceptionState
from app.services.tool_executor import ToolExecutor


logger = logging.getLogger("orchestrator")

_AVATAR_EXPRESSION_PATTERN = re.compile(
    r"\[\s*state\s*:\s*(happy|angry|sad|relaxed|surprised|neutral)\s*\]",
    re.IGNORECASE,
)
_DEFAULT_AVATAR_EXPRESSION = "neutral"
_EXPRESSION_TAG_PREFIX_PATTERN = re.compile(r"\[\s*state\s*:\s*", re.IGNORECASE)


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
        tool_executor: ToolExecutor,
        summary_trigger: int = 10,
    ):
        self.llm = llm
        self.context_builder = context_builder
        self.history = history_store
        self.memory = memory_store
        self.summary_store = summary_store
        self.summarizer = summarizer
        self.planner = planner
        self.tool_executor = tool_executor
        self.summary_trigger = summary_trigger
        self.memory_policy = memory_policy

        self.perception = PerceptionState()  # NEW

        self.session_id = str(uuid.uuid4())[:8]

        logger.info(
            "[%s] Orchestrator initialized (summary_trigger=%d)",
            self.session_id,
            summary_trigger,
        )

    def set_session(self, session_id: str):
        if self.session_id == session_id:
            return

        previous_session_id = self.session_id
        self.session_id = session_id
        self.perception = PerceptionState()

        logger.info(
            "[%s] Session activated (previous=%s)",
            self.session_id,
            previous_session_id,
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
                    tool_context = yield from self.tool_executor.execute(
                        action,
                        user_text,
                    )

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

        visible_buffer = ""
        stream_buffer = ""
        expression_initialized = False
        start_ts = time.perf_counter()

        for chunk in self.llm.stream_chat(messages):
            stream_buffer += chunk

            events, stream_buffer = self._extract_expression_events(stream_buffer)
            visible_buffer, expression_initialized = yield from self._emit_stream_events(
                events,
                visible_buffer,
                expression_initialized,
            )

        events, stream_buffer = self._extract_expression_events(stream_buffer, force=True)
        visible_buffer, expression_initialized = yield from self._emit_stream_events(
            events,
            visible_buffer,
            expression_initialized,
        )

        if not expression_initialized:
            logger.info(
                "[%s] Model selected avatar expression '%s'",
                self.session_id,
                _DEFAULT_AVATAR_EXPRESSION,
            )
            yield AvatarExpressionEvent(expression=_DEFAULT_AVATAR_EXPRESSION)

        logger.info(
            "[%s] LLM response complete (chars=%d, duration=%.2f ms)",
            self.session_id,
            len(visible_buffer),
            (time.perf_counter() - start_ts) * 1000,
        )
        return visible_buffer

    def _emit_stream_events(
        self,
        events,
        visible_buffer: str,
        expression_initialized: bool,
    ):
        for event_type, value in events:
            if event_type == "expression":
                expression_initialized = True
                logger.info(
                    "[%s] Model selected avatar expression '%s'",
                    self.session_id,
                    value,
                )
                yield AvatarExpressionEvent(expression=value)
                continue

            if not value:
                continue

            if not expression_initialized:
                expression_initialized = True
                logger.info(
                    "[%s] Model selected avatar expression '%s'",
                    self.session_id,
                    _DEFAULT_AVATAR_EXPRESSION,
                )
                yield AvatarExpressionEvent(expression=_DEFAULT_AVATAR_EXPRESSION)

            visible_buffer += value
            yield AssistantSpeechEvent(text=value)

        return visible_buffer, expression_initialized

    def _extract_expression_events(self, text: str, force: bool = False):
        events = []
        remainder = text

        while remainder:
            match = _AVATAR_EXPRESSION_PATTERN.search(remainder)
            if match:
                if match.start() > 0:
                    events.append(("text", remainder[:match.start()]))

                events.append(("expression", match.group(1).lower()))
                remainder = remainder[match.end():]
                continue

            if force:
                events.append(("text", remainder))
                return events, ""

            marker_start = self._find_incomplete_expression_start(remainder)
            if marker_start is None:
                events.append(("text", remainder))
                return events, ""

            if marker_start > 0:
                events.append(("text", remainder[:marker_start]))

            return events, remainder[marker_start:]

        return events, remainder

    def _find_incomplete_expression_start(self, text: str):
        last_bracket = text.rfind("[")
        if last_bracket == -1:
            return None

        candidate = text[last_bracket:]
        normalized = re.sub(r"\s+", "", candidate.lower())

        if "[state:".startswith(normalized):
            return last_bracket

        if _EXPRESSION_TAG_PREFIX_PATTERN.match(candidate) and "]" not in candidate:
            return last_bracket

        return None

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
