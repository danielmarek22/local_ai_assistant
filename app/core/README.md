
# Core

Core orchestration logic of the assistant.

## Responsibilities
- High-level orchestration loop
- Per-turn perception updates with an explicit `session_id`
- Memory retrieval and prompt injection
- Late-routing tool and memory directives from the model thinking stream
- Coordinating memory, tools, and LLM calls
- Streaming response/state/avatar events
- Triggering post-turn summarization

## Current Notes

- The orchestrator itself is stateless with respect to the active session.
- Session identity is supplied by the caller on each `handle_user_input(...)` call.
- `MemoryRetriever`, `MemoryActionHandler`, `TurnFinalizer`, and `StreamProcessor` keep the turn loop smaller and easier to change.
- Normal user turns use a filtered thinking stream for native `tool_call` directives; `instant_mode` bypasses that late-routing layer.

Think of this as the *spinal cord* of the system.
