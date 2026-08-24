# Integrations

Validated capabilities and optional passive context providers exposed to the assistant.

Each integration registers one or more `integration__action` capabilities through
`IntegrationRegistry`. A capability declares a JSON Schema input contract and a
handler that returns a typed `ToolResult`. The registry validates registrations at
startup and model arguments before execution.

Integrations may also return bounded observed state through `context(...)`. Passive
context is available in direct and agent modes; executable capability schemas are
sent to the model only in agent mode.

Integrations may register strict `integration__event` contracts through `EventSpec` and
publish `IntegrationEvent` instances after startup. Event payloads use JSON Schema, carry
session/correlation lineage, and execute through the durable autonomy runtime with exact
capability allowlists. Tool handlers can return `ToolResult.pending(...)`; the invocation
ID then correlates a later integration event with its originating conversation.

Built-in integrations are registered explicitly in `orchestrator_factory.py`. V1
does not dynamically import integrations from configuration or installed packages.

The optional Mindcraft integration attaches to a separately managed mindserver
and caches status and world-state events for passive context. Strict direct-action
capabilities bypass Mindcraft's planning model for supported commands, while
`mindcraft__send_message` delegates complex objectives to that model. Long-running
direct commands return pending operations and publish correlated success/failure events.
General bot output remains passive context and is not treated as command completion.

The enhanced Mindcraft fork is feature-detected through a versioned handshake. It uses
typed action requests, terminal operation recovery, exact Mineflayer events, semantic
resource actions, and verified first-person JPEG attachments. Player speech is journaled
without waking Astra by default. `controller_mode: external` disables Mindcraft model
initialization while preserving typed control, pathfinding, safety modes, and capture;
`hybrid` retains natural-language delegation.

Integrations that own background resources may implement `start(publisher)` and `close()`.
Startup occurs only after the server event loop and journal are ready; shutdown runs in
reverse registration order and isolates cleanup failures.
