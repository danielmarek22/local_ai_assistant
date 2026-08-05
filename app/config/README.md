
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
- `context.integration_context_limit` bounds passive integration state injected per turn.

Legacy `tools.web` configuration is accepted temporarily when `integrations.web` is absent.
