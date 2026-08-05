# Integrations

Validated capabilities and optional passive context providers exposed to the assistant.

Each integration registers one or more `integration__action` capabilities through
`IntegrationRegistry`. A capability declares a JSON Schema input contract and a
handler that returns a typed `ToolResult`. The registry validates registrations at
startup and model arguments before execution.

Integrations may also return bounded observed state through `context(...)`. Passive
context is available in direct and agent modes; executable capability schemas are
sent to the model only in agent mode.

Built-in integrations are registered explicitly in `orchestrator_factory.py`. V1
does not dynamically import integrations from configuration or installed packages.
