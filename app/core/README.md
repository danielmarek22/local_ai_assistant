
# Core

Core orchestration logic of the assistant.

## Responsibilities
- High-level orchestration loop
- Per-turn perception updates with an explicit `session_id`
- Memory retrieval and prompt injection
- Native late-routing capability calls in agent mode
- Coordinating memory, integrations, and LLM calls
- Bounded autonomous event turns using per-event capability allowlists
- Streaming response/state/avatar events
- Triggering post-turn summarization

## Current Notes

- The orchestrator itself is stateless with respect to the active session.
- Session identity is supplied by the caller on each `handle_user_input(...)` call.
- Conversation sessions have a durable `direct` or `manual_group` kind. Manual relay
  identities are derived from normalized sender type and display name; renaming a
  participant creates a new identity in the v1 roster-free model.
- `ContextBuilder`, `ToolExecutor`, and `TurnFinalizer` keep cross-feature turn
  coordination close to the orchestrator without placing it in feature packages.
- `ResponseGenerator` owns direct streaming, native tool-loop phases, correction and
  recovery policy, and avatar/text event emission. The orchestrator constructs it with
  explicit dependencies for each generation and retains input acceptance, authoritative
  turn identity, retrieval, assistant persistence, and finalization. Generation receives
  only a tool-trace writer, not the history store or orchestrator itself. Model/tool
  resources remain application-owned; a generator does not close them.
- `generation_step.py` represents each native inference outcome as `FinalText`,
  `AcceptedToolCall`, or `InvalidToolCall`. Invalid calls retain their raw name,
  arguments, and diagnostic for correction; accepted structure does not confer
  execution authority. The existing first-tool-call policy remains in the generator.
- `ResponseRenderer` owns one response's avatar parser, visible text, and expression
  state. Direct and buffered generation share its push/finish lifecycle, while their
  callers retain reasoning separation, group-envelope handling, and empty-response
  policy. Rendering does not execute tools or persist conversation state.
- `app/avatar/stream_processor.py` evaluates balanced bracket candidates against the configured avatar
  allowlists and removes invalid candidates. Escaped brackets and brackets inside
  fenced code, inline code, Markdown links, images, and references remain ordinary text.
- Agent-mode turns expose registered native capabilities; `instant_mode` streams a direct response without capability schemas.
- A frozen `AuthoritativeTurnContext` is created immediately after participant-message
  persistence and explicitly accompanies eligible turn-scoped tools. It contains the
  session/message, original text, observation time, sender attribution, owner, session
  kind, input source, and timezone; model arguments are not authority.
- In belief `react_tool` mode, `beliefs__update` is exposed only for eligible non-instant
  participant turns. The observer is not constructed. A hidden frozen catalog supplies
  only source-authorized invalidation IDs and is never persisted as chat.
- Native routing uses internally buffered streaming and one tool per inference. Initial
  calls and successful-tool continuations inherit the turn's normal reasoning setting;
  belief correction and tool-free recovery disable thinking and use bounded budgets.
- User and integration-event turns share a coordinator, so a local model is never used by overlapping turns.
- Autonomous final text is journaled internally; only `runtime__notify` produces a visible notification.
- The roadmap's high-level planner will be designed around validated capabilities and
  bounded observe-act-revise loops; no legacy turn planner sits on the active path.

Think of this as the *spinal cord* of the system.
