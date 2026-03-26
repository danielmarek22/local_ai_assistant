import uuid
import logging
import re
import time
import os
from typing import Generator, Optional, Dict

from app.core.events import (
    AssistantSpeechEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
)
from app.core.assistant_state import AssistantState
from app.core.actions import Action
from app.core.plan import Plan
from app.perception.state import ImageAttachment, PerceptionState
from app.services.tool_executor import ToolExecutor


logger = logging.getLogger("orchestrator")

_AVATAR_EXPRESSION_PATTERN = re.compile(
    r"\[\s*(?:state|expression)\s*:\s*(happy|angry|sad|relaxed|surprised|neutral)\s*\]",
    re.IGNORECASE,
)
_DEFAULT_AVATAR_EXPRESSION = "neutral"
_EXPRESSION_TAG_PREFIX_PATTERN = re.compile(r"\[\s*(?:state|expression)\s*:\s*", re.IGNORECASE)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL errors only


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

        self.perception = PerceptionState()

        # Initialize our dictionary to track summarized turns per session
        self._last_summary_counts: Dict[str, int] = {}

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

    def handle_user_input(
        self,
        user_text: str,
        think_override=None,
        attachments: list[ImageAttachment] | None = None,
    ):
        start_ts = time.perf_counter()
        attachments = attachments or []
        retrieval_text = self._build_retrieval_text(user_text, attachments)
        history_text = self._build_history_text(user_text, attachments)

        logger.info(
            "[%s] User input received (len=%d, images=%d)",
            self.session_id,
            len(user_text),
            len(attachments),
        )
        logger.debug("[%s] User input text: %r", self.session_id, user_text)

        yield AssistantStateEvent(state=AssistantState.THINKING)

        # --------------------------------------------------------
        # 1. Update perception with raw input
        # --------------------------------------------------------
        self.perception.update(
            "user.input",
            {
                "text": user_text,
                "source": "keyboard",
                "image_count": len(attachments),
                "attachments": [attachment.to_perception_payload() for attachment in attachments],
            },
        )

        # --------------------------------------------------------
        # 2. Vector Retrieval (Semantic + Episodic)
        # --------------------------------------------------------
        logger.debug("[%s] Querying vector DB for memories", self.session_id)
        if retrieval_text:
            semantic_memories = self.memory.get_relevant(retrieval_text, limit=3)
            episodic_memories = self.history.search_past_conversations(
                retrieval_text,
                self.session_id,
                limit=3,
            )
        else:
            semantic_memories = []
            episodic_memories = []

        memory_blocks = []
        if semantic_memories:
            memory_blocks.append("Relevant Facts:\n" + "\n".join(f"- {m}" for m in semantic_memories))
        if episodic_memories:
            memory_blocks.append("Past Conversations:\n" + "\n".join(f"- {m}" for m in episodic_memories))

        memory_context = "\n\n".join(memory_blocks) if memory_blocks else None

        # Inject memories into perception so the Planner can read them
        if memory_context:
            self.perception.update("memory.retrieved", {"value": f"\n{memory_context}\n"})
        else:
            self.perception.update("memory.retrieved", {"value": "No relevant past memories found."})

        # --------------------------------------------------------
        # 3. Persist user input (to SQLite + Vector Store)
        # --------------------------------------------------------
        self.history.add(self.session_id, "user", history_text, attachments=attachments)
        logger.debug("[%s] User input persisted to history", self.session_id)

        # --------------------------------------------------------
        # 4. Planning (decide actions)
        # --------------------------------------------------------
        perception_snapshot = self.perception.snapshot()
        plan = self._plan(retrieval_text or user_text, perception_snapshot)

        logger.debug("[%s] Plan actions: %s", self.session_id, [action.type for action in plan.actions])

        tool_context: Optional[str] = None

        # --------------------------------------------------------
        # 5. Execute actions
        # --------------------------------------------------------
        for action in plan.actions:
            logger.info("[%s] Executing action '%s'", self.session_id, action.type)

            if action.type == "web_search":
                tool_context = yield from self.tool_executor.execute(action, user_text)

            elif action.type == "write_memory":
                self._run_memory_action(action)

            elif action.type == "respond":
                logger.debug("[%s] Respond action reached, stopping action loop", self.session_id)
                break

            else:
                logger.warning("[%s] Unknown action '%s', skipping", self.session_id, action.type)

        # --------------------------------------------------------
        # 6. Context construction (Merge Memory & Tools)
        # --------------------------------------------------------
        # We combine retrieved memory and tool outputs so the context_builder 
        # doesn't need its signature changed.
        combined_context_parts = []
        if memory_context:
            combined_context_parts.append(f"--- RETRIEVED MEMORY ---\n{memory_context}")
        if tool_context:
            combined_context_parts.append(f"--- TOOL RESULTS ---\n{tool_context}")

        final_injected_context = "\n\n".join(combined_context_parts) if combined_context_parts else None

        messages = self._build_context(user_text, final_injected_context, attachments)

        # --------------------------------------------------------
        # 7. LLM streaming response
        # --------------------------------------------------------
        response = yield from self._stream_response(messages, think_override=think_override)

        # --------------------------------------------------------
        # 8. Persist assistant response (to SQLite + Vector Store)
        # --------------------------------------------------------
        self.history.add(self.session_id, "assistant", response)
        logger.debug("[%s] Assistant response persisted to history", self.session_id)

        yield AssistantSpeechEvent(text=response, is_final=True)
        yield AssistantStateEvent(state=AssistantState.IDLE)

        # --------------------------------------------------------
        # 9. Post-processing (summarization)
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

    def _plan(self, user_text: str, perception: dict) -> Plan:
        logger.info("[%s] Running planner", self.session_id)

        try:
            plan = self.planner.decide(
                user_text=user_text,
                perception=perception,
            )
        except Exception:
            logger.exception("[%s] Planner failed", self.session_id)
            raise

        logger.info("[%s] Planner produced %d actions", self.session_id, len(plan.actions))
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

    def _build_context(
        self,
        user_text: str,
        tool_context: Optional[str],
        attachments: list[ImageAttachment] | None = None,
    ):
        logger.info("[%s] Building context", self.session_id)

        messages = self.context_builder.build(
            session_id=self.session_id,
            user_text=user_text,
            injected_context=tool_context,
            attachments=attachments or [],
        )

        logger.debug(
            "[%s] Context built (messages=%d, tool_context=%s)",
            self.session_id,
            len(messages),
            bool(tool_context),
        )
        return messages

    def _build_retrieval_text(
        self,
        user_text: str,
        attachments: list[ImageAttachment],
    ) -> str:
        text = user_text.strip()
        if text:
            return text

        if not attachments:
            return ""

        names = ", ".join(
            attachment.name
            for attachment in attachments[:3]
            if attachment.name
        )
        if names:
            return f"user shared image attachments: {names}"
        return "user shared image attachments"

    def _build_history_text(
        self,
        user_text: str,
        attachments: list[ImageAttachment],
    ) -> str:
        text = user_text.strip()
        if not attachments:
            return text

        suffix = f"User attached {len(attachments)} image"
        if len(attachments) != 1:
            suffix += "s"

        if text:
            return f"{text}\n\n[{suffix}]"

        return f"[{suffix}]"

    def _stream_response(self, messages, think_override=None):
        logger.info("[%s] Calling LLM (streaming)", self.session_id)
        yield AssistantStateEvent(state=AssistantState.RESPONDING)

        visible_buffer = ""
        stream_buffer = ""
        expression_initialized = False
        start_ts = time.perf_counter()

        for chunk in self.llm.stream_chat(messages, think_override=think_override):
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
            logger.info("[%s] Model selected avatar expression '%s'", self.session_id, _DEFAULT_AVATAR_EXPRESSION)
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
                logger.info("[%s] Model selected avatar expression '%s'", self.session_id, value)
                yield AvatarExpressionEvent(expression=value)
                continue

            if not value:
                continue

            if not expression_initialized:
                expression_initialized = True
                logger.info("[%s] Model selected avatar expression '%s'", self.session_id, _DEFAULT_AVATAR_EXPRESSION)
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

        if "[state:".startswith(normalized) or "[expression:".startswith(normalized):
            return last_bracket

        if _EXPRESSION_TAG_PREFIX_PATTERN.match(candidate) and "]" not in candidate:
            return last_bracket

        return None

    # ============================================================
    # Summarization
    # ============================================================

    def _maybe_summarize(self):
        logger.debug("[%s] Checking summarization conditions", self.session_id)

        # 1. Fetch the existing summary and turn count directly from the DB
        summary_data = self.summary_store.get(self.session_id)
        if summary_data:
            existing_summary, last_count = summary_data
        else:
            existing_summary, last_count = None, 0

        # 2. Fetch history
        history = self.history.get_recent(
            session_id=self.session_id,
            limit=1000, 
        )
        current_count = len(history)

        # 3. Check if we've hit the trigger threshold
        if (current_count - last_count) < self.summary_trigger:
            return

        logger.info("[%s] Summarizing conversation history", self.session_id)
        
        summary_input = []
        if existing_summary:
            summary_input.append({
                "role": "system",
                "content": f"Here is the current summary of the conversation so far. Update it using the new messages below:\n\n{existing_summary}"
            })

        # 4. Only append the NEW messages
        summary_input.extend([
            {"role": row["role"], "content": row["content"]}
            for row in history[last_count:] 
        ])

        # 5. Generate the new summary
        try:
            summary = self.summarizer.summarize(summary_input)
        except Exception:
            logger.exception("[%s] Summarization failed", self.session_id)
            return

        # 6. Save the new summary AND the current count back to the DB
        self.summary_store.set(self.session_id, summary, current_count)

        logger.info("[%s] History summarized (%d chars)", self.session_id, len(summary))