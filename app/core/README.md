
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
- `MemoryRetriever`, `MemoryActionHandler`, `TurnFinalizer`, and `StreamProcessor` keep the turn loop smaller and easier to change.
- Agent-mode turns expose registered native capabilities; `instant_mode` streams a direct response without capability schemas.
- User and integration-event turns share a coordinator, so a local model is never used by overlapping turns.
- Autonomous final text is journaled internally; only `runtime__notify` produces a visible notification.
- The dormant planner remains available for future repurposing but is not part of the active turn path.

Think of this as the *spinal cord* of the system.
