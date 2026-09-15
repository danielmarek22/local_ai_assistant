import logging
import time
import json
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from app.core.events import (
    AssistantSpeechEvent,
    AssistantThinkingEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
    AvatarAnimationEvent,
)
from app.core.assistant_state import AssistantState
from app.avatar.stream_processor import StreamProcessor
from app.llm.thinking_filter import ThinkingBlockSplitter
from app.core.conversation import (
    InputSource,
    SenderAttribution,
    SenderType,
)
from app.logging import trace_event
from app.llm.base import InferenceFailure
from app.integrations import (
    CapabilityId,
    IntegrationEvent,
    NotificationRequest,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger("orchestrator")

_DEFAULT_AVATAR_EXPRESSION = "neutral"
SAFE_EMPTY_RESPONSE_FALLBACK = "I'm sorry, I lost my train of thought. Could you repeat that?"
_BELIEF_CAPABILITY = CapabilityId("beliefs", "update")
DEFAULT_GENERATION_DEADLINE_S = 600.0
DEFAULT_RECOVERY_DEADLINE_S = 180.0
DEFAULT_RECOVERY_NUM_PREDICT = 192
_MAX_RECOVERY_OBSERVATION_CHARS = 1200


@dataclass
class _LateRoutingTurnState:
    normal_think: object
    phase: "_GenerationPhase"
    belief_attempts: int = 0
    belief_disabled: bool = False
    tool_interactions: int = 0


class _GenerationPhase(str, Enum):
    INITIAL = "initial"
    CONTINUATION = "continuation"
    CORRECTION = "correction"
    RECOVERY = "recovery"


def inject_hidden_system_message(messages: list[dict], content: str) -> None:
    payload = {
        "role": "system",
        "content": content,
    }
    if messages and messages[0].get("role") == "system":
        messages.insert(1, payload)
    else:
        messages.insert(0, payload)


class ResponseGenerator:
    """Per-turn inference, native tool routing, recovery, and response event emission."""

    def __init__(
        self, llm, tool_executor, record_tool_trace, *, allowed_animations=None,
        allowed_expressions=None, max_late_routing_steps=5,
        generation_deadline_s=DEFAULT_GENERATION_DEADLINE_S,
        recovery_deadline_s=DEFAULT_RECOVERY_DEADLINE_S,
        recovery_num_predict=DEFAULT_RECOVERY_NUM_PREDICT,
    ):
        self.llm = llm
        self.tool_executor = tool_executor
        self.record_tool_trace = record_tool_trace
        self.allowed_animations = allowed_animations
        self.allowed_expressions = allowed_expressions
        self.max_late_routing_steps = max_late_routing_steps
        self.generation_deadline_s = generation_deadline_s
        self.recovery_deadline_s = recovery_deadline_s
        self.recovery_num_predict = recovery_num_predict

    def _inject_late_routing_system_message(self, messages: list[dict]) -> None:
        inject_hidden_system_message(
            messages,
            (
                "Internal late-routing protocol. Evaluate the user's request and the available context. "
                "If a tool action is warranted, use the available native tool schema immediately. "
                "CRITICAL: Do NOT acknowledge the user, explain what you are going to do, or use conversational "
                "filler (e.g., 'Let me check', 'One moment'). Output ONLY the native tool call. "
                "Issue at most one native tool call per inference step. If no tools are needed, "
                "answer the user directly. Before producing the final response, consider whether "
                "the current participant message warrants a tool action:\n"
                "1. Use beliefs__update, when available, for current, revisable claims about what is true.\n"
                "2. Use memory__write, when available, for events, decisions, instructions, or narrative "
                "context worth recalling in a later conversation.\n"
                "3. Normally do not store the same proposition through both tools.\n"
                "4. Use neither for incidental chat, hypotheticals, generated conclusions, Astra's "
                "opinions, or transient details without future value.\n"
                "5. The user need not explicitly say 'remember this' when durable recall value is clear.\n"
                "Examples: a stable beverage preference or revised current claim uses beliefs__update; "
                "a temporary current activity usually uses beliefs__update only; a meaningful completed "
                "shared event or durable project decision uses memory__write; incidental chat and Astra's "
                "stylistic reactions use neither. Rarely, both are justified when a current revisable fact "
                "and a distinct durable event narrative each have independent value.\n"
                "Only say information was saved or updated when a successful result from the "
                "appropriate tool in this turn confirms it. A promise to remember is not a write. "
                "If an explicit persistence request cannot be completed with the available tools, "
                "explain that limitation or ask for the missing clarification."
            ),
        )

    def stream_response(
        self,
        session_id: str,
        messages,
        think_override=None,
        options_override: dict | None = None,
        generation_deadline_s: float | None = None,
        timeout_override: float | None = None,
    ):
        logger.info("[%s] Calling LLM (streaming)", session_id)
        yield AssistantStateEvent(state=AssistantState.RESPONDING)

        visible_buffer = ""
        thinking_buffer = ""
        expression_initialized = False
        start_ts = time.perf_counter()
        processor = StreamProcessor(
            allowed_animations=self.allowed_animations,
            allowed_expressions=self.allowed_expressions,
        )
        thinking_splitter = ThinkingBlockSplitter()

        stream_kwargs = {"think_override": think_override}
        stream_parameters = inspect.signature(self.llm.stream_chat).parameters
        accepts_stream_kwargs = any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in stream_parameters.values()
        )
        if options_override is not None and (
            accepts_stream_kwargs or "options_override" in stream_parameters
        ):
            stream_kwargs["options_override"] = options_override
        if generation_deadline_s is not None and (
            accepts_stream_kwargs or "generation_deadline_s" in stream_parameters
        ):
            stream_kwargs["generation_deadline_s"] = generation_deadline_s
        if timeout_override is not None and (
            accepts_stream_kwargs or "timeout_override" in stream_parameters
        ):
            stream_kwargs["timeout_override"] = timeout_override
        if accepts_stream_kwargs or "telemetry_session_id" in stream_parameters:
            stream_kwargs["telemetry_session_id"] = session_id
        for chunk in self.llm.stream_chat(messages, **stream_kwargs):
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

    def stream_late_routed_response(
        self,
        session_id: str,
        messages,
        user_text: str,
        tool_approval_callback: Callable[[dict], bool] | None = None,
        allowed_capabilities: set[CapabilityId] | frozenset[CapabilityId] | None = None,
        event: IntegrationEvent | None = None,
        notification_callback: Callable[[NotificationRequest], bool] | None = None,
        persist_tool_traces: bool = True,
        authoritative_turn=None,
        prepared_belief_turn=None,
        initial_think_override=None,
    ):
        logger.info("[%s] Calling LLM with native late routing", session_id)
        # Freeze application authority before inference or callbacks can mutate it.
        # Event turns without explicit authority fail closed.
        execution_capabilities = (
            frozenset(allowed_capabilities) if allowed_capabilities is not None
            else frozenset() if event is not None else None
        )
        
        # THE MISSING LINK: Inject the high-level instruction before the loop
        self._inject_late_routing_system_message(messages)
        if prepared_belief_turn is not None:
            try:
                catalog_message = prepared_belief_turn.tool_catalog_message()
            except Exception:
                logger.exception(
                    "[%s] Belief mutation catalog rendering failed; tool omitted",
                    session_id,
                )
                prepared_belief_turn = None
            else:
                inject_hidden_system_message(messages, catalog_message)

        resolver = getattr(self.llm, "resolve_think_value", None)
        normal_think = (
            resolver(initial_think_override)
            if callable(resolver)
            else (True if initial_think_override is None else initial_think_override)
        )
        turn_state = _LateRoutingTurnState(
            normal_think=normal_think,
            phase=_GenerationPhase.INITIAL,
        )

        for step in range(1, self.max_late_routing_steps + 1):
            logger.info(
                "[%s] Late routing step %d/%d",
                session_id,
                step,
                self.max_late_routing_steps,
            )

            inference_phase = turn_state.phase
            try:
                result = yield from self._stream_late_routing_step(
                    session_id=session_id,
                    messages=messages,
                    user_text=user_text,
                    allowed_capabilities=execution_capabilities,
                    authoritative_turn=authoritative_turn,
                    prepared_belief_turn=prepared_belief_turn,
                    excluded_capabilities=(
                        frozenset({_BELIEF_CAPABILITY})
                        if turn_state.belief_disabled
                        else frozenset()
                    ),
                    inference_phase=inference_phase,
                    normal_think=turn_state.normal_think,
                    react_iteration=step,
                )
            except (InferenceFailure, TimeoutError, ValueError) as exc:
                category = getattr(exc, "category", type(exc).__name__)
                logger.warning(
                    "[%s] Late-routing inference failed phase=%s category=%s; recovering tool-free",
                    session_id, inference_phase.value, category,
                )
                trace_event(
                    "orchestrator", "late_routing_inference_failure",
                    session_id=session_id,
                    payload={"phase": inference_phase.value, "category": category},
                )
                return (yield from self._force_tool_free_response(
                    session_id=session_id,
                    messages=messages,
                    user_text=user_text,
                    reason=f"{inference_phase.value}_failure",
                    react_iteration=step,
                ))

            if result["tool_call"] is None and result["tool_error"] is None:
                if result["response"] and result["response"].strip():
                    return result["response"]
                reason = (
                    "empty_after_tool_interaction"
                    if turn_state.tool_interactions
                    else "initial_empty_no_tool_generation"
                )
                return (yield from self._force_tool_free_response(
                    session_id=session_id,
                    messages=messages,
                    user_text=user_text,
                    reason=reason,
                ))

            tool_call = result["tool_call"]
            tool_name = result["tool_name"]
            tool_arguments = result["tool_arguments"]
            is_belief_call = tool_name == str(_BELIEF_CAPABILITY)
            if is_belief_call:
                turn_state.belief_attempts += 1
            
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": tool_name,
                        "arguments": tool_arguments,
                    }
                }]
            })

            if is_belief_call and turn_state.belief_disabled:
                tool_result = ToolResult.error(
                    "Belief update was not applied. Do not call beliefs__update again for "
                    "this participant message. Continue with a normal response.",
                    diagnostics={
                        "category": "attempt_limit",
                        "error_code": "BELIEF_ATTEMPT_LIMIT",
                        "repository_accessed": False,
                    },
                )
            elif result["tool_error"] is not None:
                tool_result = ToolResult.error(result["tool_error"])
            else:
                tool_result = yield from self._execute_late_tool_call(
                    session_id=session_id,
                    call=tool_call,
                    user_text=user_text,
                    tool_approval_callback=tool_approval_callback,
                    allowed_capabilities=execution_capabilities,
                    event=event,
                    notification_callback=notification_callback,
                    authoritative_turn=authoritative_turn,
                    prepared_belief_turn=prepared_belief_turn,
                )
            turn_state.tool_interactions += 1
            observation = f"[{tool_result.status.value}] {tool_result.content}"

            if is_belief_call and tool_result.status.value == "error":
                diagnostics = dict(tool_result.diagnostics or {})
                trace_event(
                    "orchestrator",
                    "belief_update_rejected",
                    session_id=session_id,
                    payload={
                        "capability": tool_name,
                        "error_code": diagnostics.get("error_code", "UNCLASSIFIED_REJECTION"),
                        "category": diagnostics.get("category", "unclassified_rejection"),
                        "reason": tool_result.content[:700],
                        "argument_summary": self._bounded_belief_argument_summary(tool_arguments),
                        "repository_accessed": diagnostics.get("repository_accessed"),
                        "react_attempt": turn_state.belief_attempts,
                    },
                )
                logger.warning(
                    "[%s] beliefs__update rejected attempt=%d code=%s category=%s "
                    "repository_accessed=%s reason=%s arguments=%s",
                    session_id,
                    turn_state.belief_attempts,
                    diagnostics.get("error_code", "UNCLASSIFIED_REJECTION"),
                    diagnostics.get("category", "unclassified_rejection"),
                    diagnostics.get("repository_accessed"),
                    tool_result.content[:700],
                    json.dumps(
                        self._bounded_belief_argument_summary(tool_arguments),
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                )
                if turn_state.belief_attempts >= 2:
                    turn_state.belief_disabled = True
                    observation += (
                        "\n\nBelief update was not applied. Do not call beliefs__update "
                        "again for this participant message. Continue with a normal response."
                    )
                else:
                    turn_state.phase = _GenerationPhase.CORRECTION
            elif is_belief_call:
                turn_state.belief_disabled = True
                turn_state.phase = _GenerationPhase.CONTINUATION
            else:
                turn_state.phase = _GenerationPhase.CONTINUATION

            safe_observation = observation[:1024] + ("..." if len(observation) > 1024 else "")
            if persist_tool_traces:
                self.record_tool_trace(
                    session_id,
                    "system",
                    f"[Tool Execution Trace: {tool_name}]\n{safe_observation}",
                    sender=SenderAttribution(
                        sender_id=f"tool:{tool_name}",
                        sender_display_name=tool_name,
                        sender_type=SenderType.TOOL,
                        input_source=InputSource.TOOL_RUNTIME,
                    ),
                )

            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": (
                    f"{observation}\n\n"
                    "[SYSTEM INTERRUPT: EVALUATION PROTOCOL]\n"
                    "1. ERROR RECOVERY: If the observation contains an error, correct the precise rejected field. A rejected beliefs__update may be corrected only once.\n"
                    "2. CONTINUATION: If the data is incomplete, emit another tool call to gather more information.\n"
                    "3. CLARIFICATION: If you are stuck or need user guidance, stop calling tools and ask the user directly.\n"
                    "4. COMPLETION: If you have what you need, answer the user directly.\n"
                    "CRITICAL: Keep your internal reasoning brief and decisive. Do not output conversational filler."
                ),
            })

            if (
                is_belief_call
                and tool_result.status.value == "error"
                and turn_state.belief_attempts >= 2
            ):
                return (yield from self._force_tool_free_response(
                    session_id=session_id,
                    messages=messages,
                    user_text=user_text,
                    reason="belief_attempt_limit_reached",
                    react_iteration=step + 1,
                ))

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
        
        return (yield from self._force_tool_free_response(
            session_id=session_id,
            messages=messages,
            user_text=user_text,
            reason="late_routing_step_limit",
            react_iteration=self.max_late_routing_steps + 1,
        ))

    def _stream_late_routing_step(
        self,
        session_id: str,
        messages,
        user_text: str,
        allowed_capabilities: set[CapabilityId] | frozenset[CapabilityId] | None = None,
        authoritative_turn=None,
        prepared_belief_turn=None,
        excluded_capabilities: frozenset[CapabilityId] = frozenset(),
        inference_phase: _GenerationPhase = _GenerationPhase.INITIAL,
        normal_think=True,
        react_iteration: int = 1,
    ):
        yield AssistantStateEvent(state=AssistantState.THINKING)
        start_ts = time.perf_counter()

        # Fetch native schemas when the executor supports native discovery.
        # Keep this backward-compatible with older test doubles and wrappers.
        tools = getattr(self.tool_executor, "get_native_tools", None)
        if callable(tools):
            kwargs = {
                "session_id": session_id,
                "user_text": user_text,
                "authoritative_turn": authoritative_turn,
                "prepared_belief_turn": prepared_belief_turn,
            }
            try:
                native_tools = (
                    tools(**kwargs)
                    if allowed_capabilities is None
                    else tools(allowed_capabilities, **kwargs)
                )
            except TypeError:
                # Backward compatibility for existing executor test doubles/wrappers.
                native_tools = tools() if allowed_capabilities is None else tools(allowed_capabilities)
        else:
            native_tools = []
        if excluded_capabilities:
            excluded_names = {str(item) for item in excluded_capabilities}
            native_tools = [
                item for item in native_tools
                if item.get("function", {}).get("name") not in excluded_names
            ]

        is_correction = inference_phase is _GenerationPhase.CORRECTION
        think_value = False if is_correction else normal_think
        options_override = {"num_predict": 512} if is_correction else None
        buffered_chat = getattr(self.llm, "chat_buffered", None)
        if callable(buffered_chat):
            message = buffered_chat(
                messages=messages,
                think_override=think_value,
                options_override=options_override,
                tools=native_tools,
                timeout_override=self.generation_deadline_s,
                generation_deadline_s=self.generation_deadline_s,
                generation_phase=inference_phase.value,
                react_iteration=react_iteration,
                telemetry_session_id=session_id,
            )
        else:
            message = self.llm.chat(
                messages=messages,
                think_override=think_value,
                options_override=options_override,
                tools=native_tools,
                timeout_override=self.generation_deadline_s,
            )
        inference_duration_ms = (time.perf_counter() - start_ts) * 1000
        trace_event(
            "orchestrator",
            "late_routing_inference_timing",
            session_id=session_id,
            payload={
                "phase": inference_phase.value,
                "effective_think": think_value,
                "effective_num_predict": 512 if is_correction else None,
                "react_iteration": react_iteration,
                "duration_ms": round(inference_duration_ms, 2),
                "tools_exposed": len(native_tools),
            },
        )

        tool_call: ToolCall | None = None
        tool_error: str | None = None
        tool_name = ""
        tool_arguments: object = {}

        # 1. Yield any background thinking the model did in one chunk
        thinking_text = message.get("thinking")
        if thinking_text:
            yield AssistantThinkingEvent(text=thinking_text)

        # 2. Check for native tool calls
        if message.get("tool_calls"):
            # Gemma 4 usually only calls one tool at a time in this loop
            tc = message["tool_calls"][0]
            try:
                if not isinstance(tc, dict):
                    raise ValueError("Tool call must be an object")
                function = tc.get("function", {})
                if not isinstance(function, dict):
                    raise ValueError("Tool function must be an object")
                tool_name = function.get("name", "")
                tool_arguments = function.get("arguments", {})
                capability = CapabilityId.parse(tool_name)
                if not isinstance(tool_arguments, dict):
                    raise ValueError("Tool arguments must be an object")
                tool_call = ToolCall(capability=capability, arguments=tool_arguments)
                logger.info("[%s] Native routing selected capability '%s'", session_id, capability)
            except (TypeError, ValueError) as exc:
                tool_error = f"Invalid tool call {tool_name!r}: {exc}"
                logger.warning("[%s] %s", session_id, tool_error)

        logger.info(
            "[%s] Late-routed response step complete (duration=%.2f ms)",
            session_id,
            (time.perf_counter() - start_ts) * 1000,
        )
        
        if tool_call is not None or tool_error is not None:
            return {
                "response": "",
                "tool_call": tool_call,
                "tool_error": tool_error,
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
            }
        
        # If no tool was called, process whatever visible text it generated
        visible_content = message.get("content", "")
        clean_response = ""
        
        if visible_content:
            yield AssistantStateEvent(state=AssistantState.RESPONDING)
            
            # Re-introduce the StreamProcessor to parse avatar tags out of the raw block
            processor = StreamProcessor(
                allowed_animations=self.allowed_animations,
                allowed_expressions=self.allowed_expressions,
            )
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
            
        return {
            "response": clean_response,
            "tool_call": None,
            "tool_error": None,
            "tool_name": "",
            "tool_arguments": {},
        }

    def _force_tool_free_response(
        self,
        *,
        session_id: str,
        messages,
        user_text: str,
        reason: str,
        react_iteration: int | None = None,
    ):
        recovery_messages = self._build_instant_recovery_messages(
            messages=messages,
            user_text=user_text,
        )
        start_ts = time.perf_counter()
        failure_category = None
        try:
            response = yield from self.stream_response(
                session_id,
                recovery_messages,
                think_override=False,
                options_override={"num_predict": self.recovery_num_predict},
                generation_deadline_s=self.recovery_deadline_s,
                timeout_override=self.recovery_deadline_s,
            )
        except Exception as exc:
            failure_category = getattr(exc, "category", type(exc).__name__)
            logger.warning(
                "[%s] Tool-free recovery failed category=%s; using deterministic fallback",
                session_id, failure_category,
            )
            response = ""
        used_fallback = not response or not response.strip()
        if used_fallback:
            response = SAFE_EMPTY_RESPONSE_FALLBACK
        trace_event(
            "orchestrator",
            "forced_recovery",
            session_id=session_id,
            payload={
                "reason": reason,
                "duration_ms": round((time.perf_counter() - start_ts) * 1000, 2),
                "used_deterministic_fallback": used_fallback,
                "tools_exposed": 0,
                "failure_category": failure_category,
                "message_count": len(recovery_messages),
                "effective_think": False,
                "effective_num_predict": self.recovery_num_predict,
                "react_iteration": react_iteration,
            },
        )
        return response

    @staticmethod
    def _build_instant_recovery_messages(
        *,
        messages: list[dict],
        user_text: str,
    ) -> list[dict]:
        """Build a small prompt for a direct, no-thinking recovery stream."""
        latest_observation = ""
        for message in reversed(messages):
            if message.get("role") == "tool":
                latest_observation = str(message.get("content", ""))
                break
        if latest_observation:
            latest_observation = latest_observation[:_MAX_RECOVERY_OBSERVATION_CHARS]

        instruction = (
            "You are recovering an assistant response that did not finish in time. "
            "Answer the participant's request immediately and concisely. Do not think aloud, "
            "call tools, or mention this recovery instruction. Do not claim that an operation "
            "succeeded unless the tool observation below confirms it. If essential context is "
            "missing, say exactly what you need."
        )
        if latest_observation:
            instruction += f"\n\nLatest tool observation:\n{latest_observation}"
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_text},
        ]

    @staticmethod
    def _bounded_belief_argument_summary(arguments: object) -> dict:
        if not isinstance(arguments, dict):
            return {"argument_type": type(arguments).__name__}
        assertions = arguments.get("assertions")
        invalidations = arguments.get("invalidations")
        summary = {
            "top_level_fields": sorted(str(key)[:64] for key in arguments)[:12],
            "assertion_count": len(assertions) if isinstance(assertions, list) else None,
            "invalidation_count": len(invalidations) if isinstance(invalidations, list) else None,
        }
        if isinstance(assertions, list):
            summary["assertions"] = [
                {
                    "fields": sorted(str(key)[:64] for key in item)[:12],
                    "subject_reference": str(item.get("subject_reference", ""))[:128],
                    "predicate": str(item.get("predicate", ""))[:64],
                    "visibility": str(item.get("visibility", ""))[:32],
                    "expiry_policy": str(item.get("expiry_policy", ""))[:32],
                    "evidence_chars": len(str(item.get("evidence_excerpt", ""))),
                }
                for item in assertions[:4]
                if isinstance(item, dict)
            ]
        return summary

    def _execute_late_tool_call(
        self,
        session_id: str,
        call: ToolCall,
        user_text: str,
        tool_approval_callback: Callable[[dict], bool] | None = None,
        event: IntegrationEvent | None = None,
        notification_callback: Callable[[NotificationRequest], bool] | None = None,
        authoritative_turn=None,
        prepared_belief_turn=None,
        allowed_capabilities: frozenset[CapabilityId] | None = None,
    ):
        capability = str(call.capability)
        yield AssistantThinkingEvent(text=f"\n[Using {capability}]\n")

        try:
            execute_kwargs = {
                "session_id": session_id,
                "user_text": user_text,
                "approval_callback": tool_approval_callback,
                "authoritative_turn": authoritative_turn,
                "prepared_belief_turn": prepared_belief_turn,
                "allowed_capabilities": allowed_capabilities,
            }
            if event is not None:
                execute_kwargs.update({
                    "event_id": event.event_id,
                    "root_event_id": event.root_event_id or event.event_id,
                    "causation_id": event.causation_id,
                    "notification_callback": notification_callback,
                })
            result = yield from self.tool_executor.execute(call, **execute_kwargs)
        except Exception as exc:
            logger.exception("[%s] Late-routed tool execution failed", session_id)
            result = ToolResult.error(f"Tool execution failed: {exc}")

        trace_event(
            "orchestrator",
            "late_routing_observation",
            session_id=session_id,
            payload={
                "tool": capability,
                "status": result.status.value,
                "observation": result.content,
            },
        )
        return result

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
