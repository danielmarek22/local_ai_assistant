# Autonomy Runtime

The autonomy runtime turns validated integration events into durable, bounded agent turns.

`IntegrationEventBroker` validates and journals events before waking its worker. SQLite is
authoritative; the in-memory priority queue is only a wake-up mechanism. Pending events
survive restart, while interrupted events replay only when their `EventSpec` explicitly
declares replay safety.

`SessionTurnCoordinator` serializes model use across user and autonomous turns and gives
waiting user turns priority at turn boundaries. Autonomous turns use per-event capability
allowlists and never inherit the Web UI's instant-mode setting.

Admission checks session availability, global capacity, and user priority together under
one condition. Waiting turns reserve neither a session nor a model slot. Cancellation
removes waiting-user priority or releases active capacity, and finished sessions retain
no coordinator state. Events may wait while users remain queued; priority does not
interrupt an already active turn.

Plain model output from an event turn is an internal journal summary. The model must call
`runtime__notify` to surface text or speech unless the event policy forces or forbids a
notification. Headless events continue processing, but approval-required capabilities are
denied when no connection is available for the target session.
