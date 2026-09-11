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

Input acknowledgement delivery is independent of turn execution. The hub retains
the latest accepted message ID per session and browser, and sends it before replaying
assistant output on reconnect. It survives a failed send, successful sends with
uncertain receipt, and terminal-frame replacement. Closing a deleted session clears
this state. A `client_id` WebSocket query parameter, stored in browser session storage,
routes acknowledgement to the originating browser and its replacement connections;
other tabs may have separate pending input. Legacy clients without that parameter
share the session's legacy acknowledgement slot.

The UI persists pending message identity across refresh and ignores duplicate
acknowledgements. This is runtime reconnect recovery, not durable job recovery across
server restarts or an exactly-once input submission protocol. If browser storage is
unavailable, the client identity survives socket reconnects within the current page
only. Tool approval still requires an explicit decision; acknowledgement replay never
replays or grants approval.
