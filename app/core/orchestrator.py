import logging
import time
import os
from typing import Optional

from app.core.events import (
    AssistantSpeechEvent,
    AssistantThinkingEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
    AvatarAnimationEvent,
)
from app.core.assistant_state import AssistantState
from app.core.actions import ActionType
from app.core.plan import Plan
from app.core.stream_processor import StreamProcessor
from app.core.thinking_filter import ThinkingBlockSplitter
from app.core.turn_input import TurnInput, InputModality
from app.logging import trace_event
from app.perception.attachments import Attachment
from app.perception.state import PerceptionState
from app.services.tool_executor import ToolExecutor
from app.perception.keys import PerceptionKey

logger = logging.getLogger("orchestrator")

_DEFAULT_AVATAR_EXPRESSION = "neutral"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL errors only


class Orchestrator:
    def __init__(
        self,
        llm,
        context_builder,
        history_store,
        summary_store,
        planner,
        tool_executor: ToolExecutor,
        memory_retriever,
        memory_action_handler,
        turn_finalizer,
        gesture_catalog: dict[str, str] | None = None,
    ):
        self.llm = llm
        self.context_builder = context_builder
        self.history = history_store
        self.summary_store = summary_store
        self.planner = planner
        self.tool_executor = tool_executor
        self.memory_retriever = memory_retriever
        self.memory_action_handler = memory_action_handler
        self.turn_finalizer = turn_finalizer
        self.gesture_catalog = dict(gesture_catalog or {})
        self.allowed_animations = set(self.gesture_catalog.keys())
        self.perception = PerceptionState()

        logger.info("Orchestrator initialized")

    # ============================================================
    # Public entry point
    # ============================================================

    def handle_user_input(
        self,
        session_id: str,
        user_text: str,
        think_override=None,
        attachments: list[Attachment] | None = None,
        input_modality = InputModality.TEXT,
    ):
        start_ts = time.perf_counter()
        turn_input = TurnInput(
            user_text=user_text,
            attachments=attachments or [],
            think_override=think_override,
            input_modality=input_modality,
        )
        retrieval_text = turn_input.retrieval_text()
        history_text = turn_input.history_text()
        idle_emitted = False
        is_closing = False

        try:
            logger.info(
                "[%s] User input received (len=%d, images=%d)",
                session_id,
                len(turn_input.user_text),
                len(turn_input.attachments),
            )
            trace_event(
                "orchestrator",
                "turn_input",
                session_id=session_id,
                payload={
                    "user_text": turn_input.user_text,
                    "retrieval_text": retrieval_text,
                    "history_text": history_text,
                    "think_override": turn_input.think_override,
                    "attachments": [attachment.to_perception_payload() for attachment in turn_input.attachments],
                },
            )

            yield AssistantStateEvent(state=AssistantState.THINKING)

            # --------------------------------------------------------
            # 1. Update perception with raw input
            # --------------------------------------------------------
            self.perception.update(
                PerceptionKey.USER_INPUT,
                {
                    "text": turn_input.user_text,
                    "source": "microphone" if turn_input.input_modality == InputModality.VOICE else InputModality.TEXT,
                    "modality": turn_input.input_modality.value,
                    "image_count": len(turn_input.attachments),
                    "attachments": [
                        attachment.to_perception_payload()
                        for attachment in turn_input.attachments
                    ],
                },
            )

            # --------------------------------------------------------
            # 2. Vector Retrieval (Semantic + Episodic)
            # --------------------------------------------------------
            retrieval = self.memory_retriever.retrieve(retrieval_text, session_id)
            memory_context = retrieval.memory_context
            trace_event(
                "orchestrator",
                "memory_retrieval",
                session_id=session_id,
                payload={
                    "query": retrieval_text,
                    "memory_context": retrieval.memory_context,
                    "perception_value": retrieval.perception_value,
                },
            )
            self.perception.update(
                PerceptionKey.MEMORY_RETRIEVED,
                {"value": retrieval.perception_value},
            )

            # --------------------------------------------------------
            # 3. Persist user input (to SQLite + Vector Store)
            # --------------------------------------------------------
            self.history.add(
                session_id,
                "user",
                history_text,
                attachments=turn_input.attachments,
            )

            # --------------------------------------------------------
            # 4. Planning (decide actions)
            # --------------------------------------------------------
            perception_snapshot = self.perception.snapshot()
            plan = self._plan(
                session_id=session_id,
                user_text=retrieval_text or turn_input.user_text,
                perception=perception_snapshot,
            )

            logger.debug(
                "[%s] Plan actions: %s",
                session_id,
                [action.type for action in plan.actions],
            )
            trace_event(
                "orchestrator",
                "plan_result",
                session_id=session_id,
                payload=[
                    {"type": action.type.value, "payload": action.payload}
                    for action in plan.actions
                ],
            )

            tool_context: Optional[str] = None

            # --------------------------------------------------------
            # 5. Execute actions
            # --------------------------------------------------------
            for action in plan.actions:
                logger.info("[%s] Executing action '%s'", session_id, action.type.value)

                if action.type == ActionType.WEB_SEARCH:
                    tool_context = yield from self.tool_executor.execute(
                        action,
                        turn_input.user_text,
                    )

                elif action.type == ActionType.WRITE_MEMORY:
                    self.memory_action_handler.handle(session_id, action)

                elif action.type == ActionType.RESPOND:
                    logger.debug("[%s] Respond action reached", session_id)
                    break

                else:
                    raise ValueError(f"Orchestrator received unhandled action type: {action.type}")

            # --------------------------------------------------------
            # 6. Context construction
            # --------------------------------------------------------
            messages = self._build_context(
                session_id=session_id,
                user_text=turn_input.user_text,
                memory_context=memory_context,
                tool_context=tool_context,
                attachments=turn_input.attachments,
            )

            # --------------------------------------------------------
            # 7. LLM streaming response
            # --------------------------------------------------------
            response = yield from self._stream_response(
                session_id,
                messages,
                think_override=turn_input.think_override,
            )

            # --------------------------------------------------------
            # 8. Persist assistant response (to SQLite + Vector Store)
            # --------------------------------------------------------
            self.history.add(session_id, "assistant", response)
            trace_event(
                "orchestrator",
                "assistant_response",
                session_id=session_id,
                payload={"response": response},
            )

            yield AssistantSpeechEvent(text=response, is_final=True)

            # --------------------------------------------------------
            # 9. Post-processing (summarization)
            # --------------------------------------------------------
            self.turn_finalizer.finalize(session_id)

            logger.info(
                "[%s] Turn completed (duration=%.2f ms)",
                session_id,
                (time.perf_counter() - start_ts) * 1000,
            )

            idle_emitted = True
            yield AssistantStateEvent(state=AssistantState.IDLE)
        except GeneratorExit:
            is_closing = True
            raise
        finally:
            if not idle_emitted and not is_closing:
                idle_emitted = True
                yield AssistantStateEvent(state=AssistantState.IDLE)

    def handle_proactive_event(
        self,
        session_id: str,
        event_text: str = (
            "A local vision watchdog detected a significant visual event. "
            "Briefly and proactively help the user based on the attached frame."
        ),
        attachments: list[Attachment] | None = None,
    ):
        start_ts = time.perf_counter()
        attachments = attachments or []
        idle_emitted = False
        is_closing = False

        try:
            logger.info(
                "[%s] Proactive vision event received (images=%d)",
                session_id,
                len(attachments),
            )
            trace_event(
                "orchestrator",
                "proactive_event",
                session_id=session_id,
                payload={
                    "event_text": event_text,
                    "attachments": [
                        attachment.to_perception_payload()
                        for attachment in attachments
                    ],
                },
            )

            yield AssistantStateEvent(state=AssistantState.THINKING)

            retrieval = self.memory_retriever.retrieve(event_text, session_id)
            memory_context = retrieval.memory_context
            self.perception.update(
                PerceptionKey.MEMORY_RETRIEVED,
                {"value": retrieval.perception_value},
            )

            messages = self._build_context(
                session_id=session_id,
                user_text="",
                memory_context=memory_context,
                tool_context=None,
                attachments=attachments,
            )
            self._inject_hidden_system_message(
                messages,
                (
                    "Hidden proactive vision event. This message is system context, "
                    "not a user message and not visible to the user. "
                    f"{event_text}"
                ),
            )

            response = yield from self._stream_response(
                session_id,
                messages,
                think_override=None,
            )

            self.history.add(session_id, "assistant", response)
            trace_event(
                "orchestrator",
                "assistant_response",
                session_id=session_id,
                payload={"response": response, "source": "proactive_vision"},
            )

            yield AssistantSpeechEvent(text=response, is_final=True)
            self.turn_finalizer.finalize(session_id)

            logger.info(
                "[%s] Proactive turn completed (duration=%.2f ms)",
                session_id,
                (time.perf_counter() - start_ts) * 1000,
            )

            idle_emitted = True
            yield AssistantStateEvent(state=AssistantState.IDLE)
        except GeneratorExit:
            is_closing = True
            raise
        finally:
            if not idle_emitted and not is_closing:
                idle_emitted = True
                yield AssistantStateEvent(state=AssistantState.IDLE)

    # ============================================================
    # Planning
    # ============================================================

    def _plan(self, session_id: str, user_text: str, perception: dict) -> Plan:
        logger.info("[%s] Running planner", session_id)
        trace_event(
            "orchestrator",
            "planner_input",
            session_id=session_id,
            payload={"user_text": user_text, "perception": perception},
        )

        try:
            plan = self.planner.decide(
                user_text=user_text,
                perception=perception,
            )
        except Exception:
            logger.exception("[%s] Planner failed", session_id)
            raise

        logger.info("[%s] Planner produced %d actions", session_id, len(plan.actions))
        return plan

    # ============================================================
    # Context & response
    # ============================================================

    def _build_context(
        self,
        session_id: str,
        user_text: str,
        memory_context: Optional[str],
        tool_context: Optional[str],
        attachments: list[Attachment] | None = None,
    ):
        logger.info("[%s] Building context", session_id)

        messages = self.context_builder.build(
            session_id=session_id,
            user_text=user_text,
            memory_context=memory_context,
            tool_context=tool_context,
            attachments=attachments or [],
        )

        logger.debug(
            "[%s] Context built (messages=%d, memory=%s, tools=%s)",
            session_id,
            len(messages),
            bool(memory_context),
            bool(tool_context),
        )
        return messages

    def _inject_hidden_system_message(self, messages: list[dict], content: str) -> None:
        payload = {
            "role": "system",
            "content": content,
        }
        if messages and messages[0].get("role") == "system":
            messages.insert(1, payload)
        else:
            messages.insert(0, payload)

    def _stream_response(self, session_id: str, messages, think_override=None):
        logger.info("[%s] Calling LLM (streaming)", session_id)
        yield AssistantStateEvent(state=AssistantState.RESPONDING)

        visible_buffer = ""
        thinking_buffer = ""
        expression_initialized = False
        start_ts = time.perf_counter()
        processor = StreamProcessor(allowed_animations=self.allowed_animations)
        thinking_splitter = ThinkingBlockSplitter()

        for chunk in self.llm.stream_chat(messages, think_override=think_override):
            visible_chunk, thinking_chunk = thinking_splitter.push(chunk)
            if thinking_chunk:
                thinking_buffer += thinking_chunk
                yield AssistantThinkingEvent(text=thinking_chunk)
            if not visible_chunk:
                continue
            if not visible_buffer and not visible_chunk.strip():
                continue
            visible_buffer, expression_initialized = yield from self._emit_processor_events(
                session_id,
                processor.push(visible_chunk),
                visible_buffer,
                expression_initialized,
            )

        final_visible_chunk, final_thinking_chunk = thinking_splitter.flush()
        if final_thinking_chunk:
            thinking_buffer += final_thinking_chunk
            yield AssistantThinkingEvent(text=final_thinking_chunk)
        if final_visible_chunk:
            if not visible_buffer and not final_visible_chunk.strip():
                final_visible_chunk = ""

        if final_visible_chunk:
            visible_buffer, expression_initialized = yield from self._emit_processor_events(
                session_id,
                processor.push(final_visible_chunk),
                visible_buffer,
                expression_initialized,
            )

        visible_buffer, expression_initialized = yield from self._emit_processor_events(
            session_id,
            processor.flush(),
            visible_buffer,
            expression_initialized,
        )

        if not expression_initialized:
            logger.info("[%s] Model selected avatar expression '%s'", session_id, _DEFAULT_AVATAR_EXPRESSION)
            yield AvatarExpressionEvent(expression=_DEFAULT_AVATAR_EXPRESSION)

        logger.info(
            "[%s] LLM response complete (chars=%d, thinking_chars=%d, duration=%.2f ms)",
            session_id,
            len(visible_buffer),
            len(thinking_buffer),
            (time.perf_counter() - start_ts) * 1000,
        )
        trace_event(
            "orchestrator",
            "llm_stream_complete",
            session_id=session_id,
            payload={
                "visible_response": visible_buffer,
                "reasoning_response": thinking_buffer,
            },
        )
        return visible_buffer

    def _emit_processor_events(
        self,
        session_id: str,
        events: list[tuple[str, str]],
        visible_buffer: str,
        expression_initialized: bool,
    ):
        for event_type, value in events:
            if event_type == "expression":
                expression_initialized = True
                logger.info("[%s] Model selected avatar expression '%s'", session_id, value)
                yield AvatarExpressionEvent(expression=value)
                continue

            if event_type == "animation":
                logger.info("[%s] Model selected avatar animation '%s'", session_id, value)
                yield AvatarAnimationEvent(animation=value)
                continue

            if not value:
                continue

            if not expression_initialized:
                expression_initialized = True
                logger.info("[%s] Model selected avatar expression '%s'", session_id, _DEFAULT_AVATAR_EXPRESSION)
                yield AvatarExpressionEvent(expression=_DEFAULT_AVATAR_EXPRESSION)

            visible_buffer += value
            yield AssistantSpeechEvent(text=value)

        return visible_buffer, expression_initialized
