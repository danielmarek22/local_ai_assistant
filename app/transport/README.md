# Transport

Wire contracts and live connection coordination for external clients.

## Responsibilities

- Validate inbound and outbound WebSocket frames
- Enforce per-frame size and sequencing rules
- Serialize sends and track connections, turns, and pending approvals
- Preserve messages received while an approval decision is pending

Transport may translate client frames into application inputs and encode application
events for clients. Domain, memory, perception, and integration code should not depend
on live WebSocket objects or transport-specific send/receive behavior.

Approval decision parsing remains with the connection hub because it is part of the
WebSocket request/response lifecycle rather than a standalone domain service.
