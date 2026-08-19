
# Config

Configuration management for the assistant.

## Responsibilities
- Loading YAML configuration files
- Providing structured access to config sections (LLM, assistant, integrations, etc.)
- Acting as a single source of truth for runtime parameters

## Key Files
- `assistant.yaml` – main configuration file
- `config.py` – config loader and accessor logic

## LLM Generation
- `llm.generation.temperature` - sampling temperature
- `llm.generation.top_p` - nucleus sampling cutoff
- `llm.generation.top_k` - top-k sampling cutoff forwarded to Ollama as `top_k`
- `llm.generation.rep_pen` - repetition penalty forwarded to Ollama as `repeat_penalty`
- `llm.generation.max_tokens` - maximum generated tokens, forwarded to Ollama as `num_predict`

This layer should remain *dumb*: no business logic, only structured data.

## Integrations

- `integrations.web` configures the optional SearXNG-backed search capability.
- `integrations.memory` enables structured long-term memory writes.
- `integrations.shell` enables local shell execution and its approval policy.
- `integrations.mindcraft` attaches to a running Mindcraft mindserver. It can
  provide cached world state, strict direct actions, and complex task delegation.
- `context.integration_context_limit` bounds passive integration state injected per turn.
- `autonomy` configures durable event processing, global model concurrency, causal-chain
  limits, queue size, approvals, and recent autonomous context.

An empty `integrations.mindcraft.agent_name` auto-selects the agent only when
exactly one connected, in-game agent is available. Set it explicitly for
multi-agent Mindcraft servers. Mindcraft runs independently; the assistant does
not start, stop, or restart its processes. Initial connection failures use
exponential backoff up to `integrations.mindcraft.reconnect_max_delay_s` and log
one warning per outage.
`integrations.mindcraft.events_enabled` enables correlated command completion events.
`integrations.mindcraft.ambient_session_id` is the explicit target for spontaneous events;
an empty value journals unbound events without invoking the assistant.
`integrations.mindcraft.autonomous_events` lists exact game events allowed to start bounded
turns after a session controls the bot. `integrations.mindcraft.attachment_dir` stores
verified first-person captures received from the enhanced fork.

Autonomy is globally pausable through `/api/autonomy` and the Web UI. Pausing leaves queued
events durable and does not cancel an already-running external action.

Legacy `tools.web` configuration is accepted temporarily when `integrations.web` is absent.
