import logging
import time
import os
from typing import Optional

from app.core.actions import Action, ActionType
from app.core.events import (
    AssistantSpeechEvent,
    AssistantThinkingEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
    AvatarAnimationEvent,
)
from app.core.assistant_state import AssistantState
from app.core.stream_processor import StreamProcessor
from app.core.thinking_filter import ThinkingBlockSplitter, ThinkingDirectiveFilter, ThinkingDirective
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
        self.tool_executor = tool_executor
        self.memory_retriever = memory_retriever
        self.memory_action_handler = memory_action_handler
        self.turn_finalizer = turn_finalizer
        self.gesture_catalog = dict(gesture_catalog or {})
        self.allowed_animations = set(self.gesture_catalog.keys())
        self.perception = PerceptionState()
        self.max_late_routing_steps = 5

        logger.info("Orchestrator initialized with late routing")

    # ============================================================
    # Public entry point
    # ============================================================

    def handle_user_input(
        self,
        session_id: str,
        user_text: str,
        think_override=None,
        instant_mode: bool = False,
        attachments: list[Attachment] | None = None,
        input_modality = InputModality.TEXT,
    ):
        start_ts = time.perf_counter()
        turn_input = TurnInput(
            user_text=user_text,
            attachments=attachments or [],
            think_override=think_override,
            instant_mode=instant_mode,
            input_modality=input_modality,
        )
        retrieval_text = turn_input.retrieval_text()
        history_text = turn_input.history_text()
        idle_emitted = False
        is_closing = False

        try:
            logger.info(
                "[%s] User input received (len=%d, images=%d, instant=%s)",
                session_id,
                len(turn_input.user_text),
                len(turn_input.attachments),
                turn_input.instant_mode,
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
                    "instant_mode": turn_input.instant_mode,
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
            # 4. Context construction
            # --------------------------------------------------------
            messages = self._build_context(
                session_id=session_id,
                user_text=turn_input.user_text,
                memory_context=memory_context,
                tool_context=None,
                attachments=turn_input.attachments,
            )

            # --------------------------------------------------------
            # 5. LLM streaming response with late routing
            # --------------------------------------------------------
            if turn_input.instant_mode:
                logger.info("[%s] Instant mode enabled; late routing disabled", session_id)
                response = yield from self._stream_response(
                    session_id,
                    messages,
                    think_override=turn_input.think_override,
                )
            else:
                response = yield from self._stream_late_routed_response(
                    session_id=session_id,
                    messages=messages,
                    user_text=turn_input.user_text,
                )

            self.history.add(session_id, "assistant", response)
            trace_event(
                "orchestrator",
                "assistant_response",
                session_id=session_id,
                payload={"response": response},
            )

            yield AssistantSpeechEvent(text=response, is_final=True)

            # --------------------------------------------------------
            # 6. Post-processing (summarization)
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

            # Proactive events skip the router and go straight to fast chat
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
        directive_filter = ThinkingDirectiveFilter()

        for chunk in self.llm.stream_chat(messages, think_override=think_override):
            visible_chunk, thinking_chunk = thinking_splitter.push(chunk)
            if thinking_chunk:
                thinking_buffer += thinking_chunk
                yield AssistantThinkingEvent(text=thinking_chunk)
            if not visible_chunk:
                continue
            
            # Sanitize visible chunk in case directive tags leaked outside thinking blocks
            visible_chunk = directive_filter.strip_directive_tags(visible_chunk)
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
            final_visible_chunk = directive_filter.strip_directive_tags(final_visible_chunk)
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

    def _stream_late_routed_response(
        self,
        session_id: str,
        messages,
        user_text: str,
    ):
        logger.info("[%s] Calling LLM with late routing", session_id)
        self._inject_late_routing_system_message(messages)

        for step in range(1, self.max_late_routing_steps + 1):
            logger.info(
                "[%s] Late routing step %d/%d",
                session_id,
                step,
                self.max_late_routing_steps,
            )

            result = yield from self._stream_late_routing_step(
                session_id=session_id,
                messages=messages,
                user_text=user_text,
            )

            if result["tool_action"] is None:
                # Check if there was a directive error to feed back
                if "directive_error" in result:
                    # Feed the error back and let the model try again
                    observation = (
                        f"Your previous tool directive was invalid. Error: {result['directive_error']}. "
                        "Please try again with correct format or provide a direct answer."
                    )
                    logger.info("[%s] Feeding back directive error (step %d)", session_id, step)
                    messages.append({
                        "role": "user",
                        "content": f"Internal observation: {observation}",
                    })
                    continue
                
                return result["response"]

            action = result["tool_action"]
            observation = yield from self._execute_late_tool_action(
                session_id=session_id,
                action=action,
                user_text=user_text,
            )
            messages.append({
                "role": "user",
                "content": (
                    f"Internal observation from {action.type.value}. "
                    "Use this to answer the original user request. "
                    "Do not mention this routing protocol.\n\n"
                    f"{observation}"
                ),
            })

        logger.warning("[%s] Late routing hit max steps; forcing final answer", session_id)
        messages.append({
            "role": "user",
            "content": (
                "Internal instruction: stop calling tools and answer the original "
                "user request with the information already available."
            ),
        })
        response = yield from self._stream_response(
            session_id,
            messages,
            think_override=True,
        )
        return response

    def _stream_late_routing_step(
        self,
        session_id: str,
        messages,
        user_text: str,
    ):
        yield AssistantStateEvent(state=AssistantState.THINKING)

        visible_buffer = ""
        thinking_buffer = ""
        expression_initialized = False
        tool_action: Action | None = None
        directive_errors: list[str] = []
        start_ts = time.perf_counter()
        processor = StreamProcessor(allowed_animations=self.allowed_animations)
        thinking_splitter = ThinkingBlockSplitter()
        directive_filter = ThinkingDirectiveFilter()
        visible_directive_filter = ThinkingDirectiveFilter()  # Separate filter for visible content

        for chunk in self.llm.stream_chat(messages, think_override=True):
            visible_chunk, thinking_chunk = thinking_splitter.push(chunk)

            if thinking_chunk:
                clean_thinking, directives = directive_filter.push(thinking_chunk)
                thinking_buffer += clean_thinking
                if clean_thinking:
                    yield AssistantThinkingEvent(text=clean_thinking)

                tool_action, errors = self._handle_late_routing_directives(
                    session_id=session_id,
                    directives=directives,
                )
                directive_errors.extend(errors)
                if tool_action is not None:
                    # Tool action detected; discard any accumulated visible content
                    # and stop streaming
                    visible_buffer = ""
                    break

            if visible_chunk:
                # Extract directives from visible chunk (model may emit them outside <think>)
                clean_visible, visible_directives = visible_directive_filter.push(visible_chunk)
                
                # Handle directives found in visible content
                if visible_directives and tool_action is None:
                    tool_action, errors = self._handle_late_routing_directives(
                        session_id=session_id,
                        directives=visible_directives,
                    )
                    directive_errors.extend(errors)
                    if tool_action is not None:
                        visible_buffer = ""
                        break
                
                if not clean_visible:
                    continue
                    
                if not visible_buffer:
                    yield AssistantStateEvent(state=AssistantState.RESPONDING)
                visible_buffer, expression_initialized = yield from self._emit_processor_events(
                    session_id,
                    processor.push(clean_visible),
                    visible_buffer,
                    expression_initialized,
                )

        final_visible_chunk, final_thinking_chunk = thinking_splitter.flush()
        if final_thinking_chunk and tool_action is None:
            clean_thinking, directives = directive_filter.push(final_thinking_chunk)
            thinking_buffer += clean_thinking
            if clean_thinking:
                yield AssistantThinkingEvent(text=clean_thinking)
            tool_action, errors = self._handle_late_routing_directives(
                session_id=session_id,
                directives=directives,
            )
            directive_errors.extend(errors)

        clean_thinking, directives = directive_filter.flush()
        if clean_thinking and tool_action is None:
            thinking_buffer += clean_thinking
            yield AssistantThinkingEvent(text=clean_thinking)
        if tool_action is None:
            tool_action, errors = self._handle_late_routing_directives(
                session_id=session_id,
                directives=directives,
            )
            directive_errors.extend(errors)

        if tool_action is not None:
            logger.info(
                "[%s] Late routing selected tool '%s'",
                session_id,
                tool_action.type.value,
            )
            return {"response": "", "tool_action": tool_action}

        if final_visible_chunk:
            # Extract any directives from final visible chunk
            clean_final_visible, final_visible_directives = visible_directive_filter.push(final_visible_chunk)
            if final_visible_directives and tool_action is None:
                tool_action, errors = self._handle_late_routing_directives(
                    session_id=session_id,
                    directives=final_visible_directives,
                )
                directive_errors.extend(errors)
                if tool_action is not None:
                    logger.info(
                        "[%s] Late routing selected tool '%s' from final chunk",
                        session_id,
                        tool_action.type.value,
                    )
                    return {"response": "", "tool_action": tool_action}
            
            if clean_final_visible:
                if not visible_buffer:
                    yield AssistantStateEvent(state=AssistantState.RESPONDING)
                visible_buffer, expression_initialized = yield from self._emit_processor_events(
                    session_id,
                    processor.push(clean_final_visible),
                    visible_buffer,
                    expression_initialized,
                )
        
        # Flush visible directive filter
        clean_remaining_visible, remaining_visible_directives = visible_directive_filter.flush()
        if remaining_visible_directives and tool_action is None:
            tool_action, errors = self._handle_late_routing_directives(
                session_id=session_id,
                directives=remaining_visible_directives,
            )
            directive_errors.extend(errors)
            if tool_action is not None:
                logger.info(
                    "[%s] Late routing selected tool '%s' from remaining visible directives",
                    session_id,
                    tool_action.type.value,
                )
                return {"response": "", "tool_action": tool_action}
        
        if clean_remaining_visible:
            if not visible_buffer:
                yield AssistantStateEvent(state=AssistantState.RESPONDING)
            visible_buffer, expression_initialized = yield from self._emit_processor_events(
                session_id,
                processor.push(clean_remaining_visible),
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
            "[%s] Late-routed response complete (chars=%d, thinking_chars=%d, duration=%.2f ms)",
            session_id,
            len(visible_buffer),
            len(thinking_buffer),
            (time.perf_counter() - start_ts) * 1000,
        )
        trace_event(
            "orchestrator",
            "late_routing_stream_complete",
            session_id=session_id,
            payload={
                "visible_response": visible_buffer,
                "filtered_thinking_response": thinking_buffer,
            },
        )
        
        # If there were directive errors and no response, treat it as a failed tool attempt
        if directive_errors and not visible_buffer:
            observation = "\n".join(directive_errors)
            return {"response": "", "tool_action": None, "directive_error": observation}
        
        return {"response": visible_buffer, "tool_action": None}

    def _handle_late_routing_directives(
        self,
        session_id: str,
        directives: list[ThinkingDirective],
    ) -> tuple[Action | None, list[str]]:
        """
        Handle late-routing directives. Returns a tuple of (tool_action, error_messages).
        error_messages contains any issues that occurred during directive processing
        that should be fed back to the model.
        """
        tool_action = None
        error_messages = []
        
        for directive in directives:
            trace_event(
                "orchestrator",
                "late_routing_directive",
                session_id=session_id,
                payload={"kind": directive.kind, "payload": directive.payload},
            )
            if directive.kind == "memory_write":
                action = Action(type=ActionType.WRITE_MEMORY, payload=directive.payload)
                self.memory_action_handler.handle(session_id, action)
                continue

            if directive.kind == "tool_call":
                tool_action = self._action_from_tool_directive(directive.payload)
                if tool_action is None:
                    error_msg = (
                        f"Invalid tool directive format: {directive.raw}. "
                        "Expected format: {\"tool\":\"web_search\",\"kwargs\":{{...}}} or "
                        "{\"name\":\"tool_name\",\"arguments\":{{...}}}"
                    )
                    logger.warning("[%s] %s", session_id, error_msg)
                    error_messages.append(error_msg)
        
        return tool_action, error_messages

    def _action_from_tool_directive(self, payload: dict) -> Action | None:
        tool_name = payload.get("tool") or payload.get("name")
        kwargs = payload.get("kwargs") or payload.get("arguments") or {}
        if not isinstance(kwargs, dict):
            kwargs = {}

        if tool_name == ActionType.WEB_SEARCH.value:
            return Action(type=ActionType.WEB_SEARCH, payload=kwargs)

        return None

    def _execute_late_tool_action(
        self,
        session_id: str,
        action: Action,
        user_text: str,
    ):
        yield AssistantStateEvent(state=AssistantState.SEARCHING)
        yield AssistantThinkingEvent(text=f"\n[Using {action.type.value}]\n")

        try:
            observation = yield from self.tool_executor.execute(action, user_text)
        except Exception as exc:
            logger.exception("[%s] Late-routed tool execution failed", session_id)
            observation = f"Tool execution failed: {exc}"

        if not observation:
            observation = "The tool returned no usable results."

        trace_event(
            "orchestrator",
            "late_routing_observation",
            session_id=session_id,
            payload={
                "tool": action.type.value,
                "observation": observation,
            },
        )
        return observation

    def _inject_late_routing_system_message(self, messages: list[dict]) -> None:
        tool_lines = []
        for name, tool in sorted(self.tool_executor.tools.items()):
            if getattr(tool, "is_available", False):
                tool_lines.append(f"- {name}")
        available_tools = "\n".join(tool_lines) if tool_lines else "- No tools are currently available."

        self._inject_hidden_system_message(
            messages,
            (
                "Internal late-routing protocol. Use this protocol only inside "
                "your private <think> block; never reveal these tags or JSON to "
                "the user.\n\n"
                "Available tools:\n"
                f"{available_tools}\n\n"
                "If you need a tool before answering, emit exactly one directive "
                "inside thinking and then wait for the observation:\n"
                '<tool_call>{"tool":"web_search","kwargs":{"query":"search terms"}}</tool_call>\n\n'
                "If the user explicitly asks you to remember something enduring, "
                "emit this inside thinking and still answer normally:\n"
                '<memory_write>{"content":"fact to remember","category":"general","importance":2}</memory_write>\n\n'
                "If no tool or memory write is needed, do not emit directives."
            ),
        )

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
