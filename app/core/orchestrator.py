import logging
import time
import os
from typing import Callable, Optional

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
from app.core.thinking_filter import ThinkingBlockSplitter
from app.core.turn_input import TurnInput, InputModality
from app.logging import trace_event
from app.perception.attachments import Attachment, ImageAttachment
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
        turn_finalizer,
        gesture_catalog: dict[str, str] | None = None,
    ):
        self.llm = llm
        self.context_builder = context_builder
        self.history = history_store
        self.summary_store = summary_store
        self.tool_executor = tool_executor
        self.memory_retriever = memory_retriever
        self.turn_finalizer = turn_finalizer
        self.gesture_catalog = dict(gesture_catalog or {})
        self.allowed_animations = set(self.gesture_catalog.keys())
        self.perception = PerceptionState()
        self.max_late_routing_steps = 5

        logger.info("Orchestrator initialized with native late routing")

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
        tool_approval_callback: Callable[[dict], bool] | None = None,
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

            # 1. Update perception with raw input
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

            # 2. Vector Retrieval (Semantic + Episodic)
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

            # 3. Persist user input (to SQLite + Vector Store)
            storable_attachments = [
                attachment
                for attachment in turn_input.attachments
                if isinstance(attachment, ImageAttachment)
            ]
            self.history.add(
                session_id,
                "user",
                history_text,
                attachments=storable_attachments,
            )

            # 4. Context construction
            messages = self._build_context(
                session_id=session_id,
                user_text=turn_input.user_text,
                memory_context=memory_context,
                tool_context=None,
                attachments=turn_input.attachments,
            )

            # 5. LLM streaming response with native late routing
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
                    tool_approval_callback=tool_approval_callback,
                )

            # Prevent History Poisoning by dropping empty responses
            if response and response.strip():
                self.history.add(session_id, "assistant", response)
                trace_event(
                    "orchestrator",
                    "assistant_response",
                    session_id=session_id,
                    payload={"response": response},
                )
                yield AssistantSpeechEvent(text=response, is_final=True)
            else:
                logger.warning("[%s] LLM returned empty response. Dropping to prevent history poisoning.", session_id)
                fallback = "I'm sorry, I lost my train of thought. Could you repeat that?"
                yield AssistantSpeechEvent(text=fallback, is_final=True)

            # 6. Post-processing (summarization)
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
    
    def _inject_late_routing_system_message(self, messages: list[dict]) -> None:
        self._inject_hidden_system_message(
            messages,
            (
                "Internal late-routing protocol. Evaluate the user's request and the available context. "
                "If you need external information, a command, or a structured memory write, you MUST immediately use the "
                "available tools provided in the native system schema. "
                "CRITICAL: Do NOT acknowledge the user, explain what you are going to do, or use conversational "
                "filler (e.g., 'Let me check', 'One moment'). Output ONLY the native tool call. "
                "If no tools are needed, answer the user directly."
            ),
        )

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
            text_chunk = chunk.get("content", "") if isinstance(chunk, dict) else chunk
            if not text_chunk:
                continue

            visible_chunk, thinking_chunk = thinking_splitter.push(text_chunk)
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
        return visible_buffer

    def _stream_late_routed_response(
        self,
        session_id: str,
        messages,
        user_text: str,
        tool_approval_callback: Callable[[dict], bool] | None = None,
    ):
        logger.info("[%s] Calling LLM with native late routing", session_id)
        
        # THE MISSING LINK: Inject the high-level instruction before the loop
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
                return result["response"]

            action = result["tool_action"]
            
            # 1. Save the tool call intent to the messages array FIRST
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": action.type.value,
                        "arguments": action.payload
                    }
                }]
            })

            # 2. Execute the tool (THIS DEFINES THE 'observation' VARIABLE)
            observation = yield from self._execute_late_tool_action(
                session_id=session_id,
                action=action,
                user_text=user_text,
                tool_approval_callback=tool_approval_callback,
            )

            # 3. Save the trace to your history log
            safe_observation = observation[:1024] + ("..." if len(observation) > 1024 else "")
            self.history.add(
                session_id, 
                "system",
                f"[Tool Execution Trace: {action.type.value}]\n{safe_observation}"
            )

            # 4. Feed the observation and protocol back to the model
            messages.append({
                "role": "tool",
                "tool_name": action.type.value,
                "content": (
                    f"{observation}\n\n"
                    "[SYSTEM INTERRUPT: EVALUATION PROTOCOL]\n"
                    "1. ERROR RECOVERY: If the observation contains an error, DO NOT apologize. Emit a NEW tool call with corrected parameters, or try a different approach.\n"
                    "2. CONTINUATION: If the data is incomplete, emit another tool call to gather more information.\n"
                    "3. CLARIFICATION: If you are stuck or need user guidance, stop calling tools and ask the user directly.\n"
                    "4. COMPLETION: If you have what you need, answer the user directly.\n"
                    "CRITICAL: Keep your internal reasoning brief and decisive. Do not output conversational filler."
                ),
            })

        # --- LOOP EXHAUSTION FALLBACK ---
        logger.warning("[%s] Late routing hit max steps; forcing final answer", session_id)
        messages.append({
            "role": "user",
            "content": (
                "[SYSTEM INTERRUPT: MAX TOOL STEPS REACHED]\n"
                "You have reached the maximum allowed tool calls for this turn. "
                "Stop calling tools. Answer the original user request based ONLY on the information you have gathered so far. "
                "If you were unable to complete the task, explicitly explain what you tried, why it failed, and what you need from the user to proceed."
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
        start_ts = time.perf_counter()

        # Fetch native schemas when the executor supports native discovery.
        # Keep this backward-compatible with older test doubles and wrappers.
        tools = getattr(self.tool_executor, "get_native_tools", None)
        native_tools = tools() if callable(tools) else []

        # Perform a BLOCKING call to guarantee the tool_calls object is returned safely
        message = self.llm.chat(
            messages=messages, 
            think_override=True, 
            tools=native_tools,
            timeout_override=120.0,  # Ensure the LLM call doesn't hang indefinitely
        )

        tool_action: Action | None = None

        # 1. Yield any background thinking the model did in one chunk
        thinking_text = message.get("thinking")
        if thinking_text:
            yield AssistantThinkingEvent(text=thinking_text)

        # 2. Check for native tool calls
        if message.get("tool_calls"):
            # Gemma 4 usually only calls one tool at a time in this loop
            tc = message["tool_calls"][0]
            func_name = tc["function"]["name"]
            kwargs = tc["function"]["arguments"]
            
            try:
                tool_action = Action(type=ActionType(func_name), payload=kwargs)
                logger.info("[%s] Native routing selected tool '%s'", session_id, tool_action.type.value)
            except ValueError:
                logger.warning("[%s] Unknown native tool called: %s", session_id, func_name)

        logger.info(
            "[%s] Late-routed response step complete (duration=%.2f ms)",
            session_id,
            (time.perf_counter() - start_ts) * 1000,
        )
        
        if tool_action is not None:
            return {"response": "", "tool_action": tool_action}
        
        # If no tool was called, process whatever visible text it generated
        visible_content = message.get("content", "")
        clean_response = ""
        
        if visible_content:
            yield AssistantStateEvent(state=AssistantState.RESPONDING)
            
            # Re-introduce the StreamProcessor to parse avatar tags out of the raw block
            processor = StreamProcessor(allowed_animations=self.allowed_animations)
            expression_initialized = False
            
            # Push the text through the processor to strip tags and yield animation events
            clean_response, expression_initialized = yield from self._emit_processor_events(
                session_id,
                processor.push(visible_content),
                "",
                expression_initialized,
            )
            
            # Flush any remaining text in the processor's buffer
            clean_response, expression_initialized = yield from self._emit_processor_events(
                session_id,
                processor.flush(),
                clean_response,
                expression_initialized,
            )

            # Ensure a default expression is set if the model didn't provide one
            if not expression_initialized:
                logger.info("[%s] Model selected avatar expression '%s'", session_id, _DEFAULT_AVATAR_EXPRESSION)
                yield AvatarExpressionEvent(expression=_DEFAULT_AVATAR_EXPRESSION)
            
        return {"response": clean_response, "tool_action": None}

    def _execute_late_tool_action(
        self,
        session_id: str,
        action: Action,
        user_text: str,
        tool_approval_callback: Callable[[dict], bool] | None = None,
    ):
        yield AssistantStateEvent(state=AssistantState.SEARCHING)
        yield AssistantThinkingEvent(text=f"\n[Using {action.type.value}]\n")

        try:
            observation = yield from self.tool_executor.execute(
                action,
                user_text,
                approval_callback=tool_approval_callback,
            )
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

            if not value or not value.strip():
                continue

            if not expression_initialized:
                expression_initialized = True
                logger.info("[%s] Model selected avatar expression '%s'", session_id, _DEFAULT_AVATAR_EXPRESSION)
                yield AvatarExpressionEvent(expression=_DEFAULT_AVATAR_EXPRESSION)

            visible_buffer += value
            yield AssistantSpeechEvent(text=value)

        return visible_buffer, expression_initialized
