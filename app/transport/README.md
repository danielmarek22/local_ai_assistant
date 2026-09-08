# Transport

Wire contracts and live connection coordination for external clients.

## Responsibilities

- Validate inbound and outbound WebSocket frames
- Enforce per-frame size and sequencing rules
- Serialize sends and track connections, turns, pending approvals, and session-visible
  assistant state
- Preserve messages received while an approval decision is pending

Transport may translate client frames into application inputs and encode application
events for clients. Domain, memory, perception, and integration code should not depend
on live WebSocket objects or transport-specific send/receive behavior.

Approval decision parsing remains with the connection hub because it is part of the
WebSocket request/response lifecycle rather than a standalone domain service.

Assistant state is runtime state rather than durable chat history. The connection hub
owns its session-scoped snapshot so refreshed clients can restore the current state and
active turn from `session_init`; completed or interrupted turns reset that snapshot.
Assistant output is broadcast to live connections for the session. If every connection
is unavailable during a short refresh gap, a bounded frame buffer preserves the active
partial stream or its terminal full response for the next connection.
