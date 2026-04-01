import logging
import time
import os
from typing import Optional

from app.core.events import (
    AssistantSpeechEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
)
from app.core.assistant_state import AssistantState
from app.core.actions import ActionType
from app.core.plan import Plan
from app.perception.state import ImageAttachment, PerceptionState
from app.core.stream_processor import StreamProcessor
from app.core.turn_input import TurnInput
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

        logger.info("Orchestrator initialized")

    # ============================================================
    # Public entry point
    # ============================================================

    def handle_user_input(
        self,
        session_id: str,
        user_text: str,
        think_override=None,
        attachments: list[ImageAttachment] | None = None,
    ):
        start_ts = time.perf_counter()
        perception = PerceptionState()
        turn_input = TurnInput(
            user_text=user_text,
            attachments=attachments or [],
            think_override=think_override,
        )
        retrieval_text = turn_input.retrieval_text()
        history_text = turn_input.history_text()
        idle_emitted = False

        try:
            logger.info(
                "[%s] User input received (len=%d, images=%d)",
                session_id,
                len(turn_input.user_text),
                len(turn_input.attachments),
            )
            logger.debug("[%s] User input text: %r", session_id, turn_input.user_text)

            yield AssistantStateEvent(state=AssistantState.THINKING)

            # --------------------------------------------------------
            # 1. Update perception with raw input
            # --------------------------------------------------------
            perception.update(
                PerceptionKey.USER_INPUT,
                {
                    "text": turn_input.user_text,
                    "source": "keyboard",
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
            perception.update(
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
            logger.debug("[%s] User input persisted to history", session_id)

            # --------------------------------------------------------
            # 4. Planning (decide actions)
            # --------------------------------------------------------
            perception_snapshot = perception.snapshot()
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
                    logger.debug("[%s] Respond action reached, stopping action loop", session_id)
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
            logger.debug("[%s] Assistant response persisted to history", session_id)

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
        finally:
            if not idle_emitted:
                idle_emitted = True
                yield AssistantStateEvent(state=AssistantState.IDLE)

    # ============================================================
    # Planning
    # ============================================================

    def _plan(self, session_id: str, user_text: str, perception: dict) -> Plan:
        logger.info("[%s] Running planner", session_id)

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
        attachments: list[ImageAttachment] | None = None,
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

    def _stream_response(self, session_id: str, messages, think_override=None):
        logger.info("[%s] Calling LLM (streaming)", session_id)
        yield AssistantStateEvent(state=AssistantState.RESPONDING)

        visible_buffer = ""
        expression_initialized = False
        start_ts = time.perf_counter()
        processor = StreamProcessor()

        for chunk in self.llm.stream_chat(messages, think_override=think_override):
            for event_type, value in processor.push(chunk):
                if event_type == "expression":
                    expression_initialized = True
                    logger.info("[%s] Model selected avatar expression '%s'", session_id, value)
                    yield AvatarExpressionEvent(expression=value)
                    continue

                if not value:
                    continue

                if not expression_initialized:
                    expression_initialized = True
                    logger.info("[%s] Model selected avatar expression '%s'", session_id, _DEFAULT_AVATAR_EXPRESSION)
                    yield AvatarExpressionEvent(expression=_DEFAULT_AVATAR_EXPRESSION)

                visible_buffer += value
                yield AssistantSpeechEvent(text=value)

        for event_type, value in processor.flush():
            if event_type == "expression":
                expression_initialized = True
                logger.info("[%s] Model selected avatar expression '%s'", session_id, value)
                yield AvatarExpressionEvent(expression=value)
                continue

            if not value:
                continue

            if not expression_initialized:
                expression_initialized = True
                logger.info("[%s] Model selected avatar expression '%s'", session_id, _DEFAULT_AVATAR_EXPRESSION)
                yield AvatarExpressionEvent(expression=_DEFAULT_AVATAR_EXPRESSION)

            visible_buffer += value
            yield AssistantSpeechEvent(text=value)

        if not expression_initialized:
            logger.info("[%s] Model selected avatar expression '%s'", session_id, _DEFAULT_AVATAR_EXPRESSION)
            yield AvatarExpressionEvent(expression=_DEFAULT_AVATAR_EXPRESSION)

        logger.info(
            "[%s] LLM response complete (chars=%d, duration=%.2f ms)",
            session_id,
            len(visible_buffer),
            (time.perf_counter() - start_ts) * 1000,
        )
        return visible_buffer
