
# Config

Configuration management for the assistant.

## Responsibilities
- Loading YAML configuration files
- Providing structured access to config sections (LLM, assistant, tools, etc.)
- Acting as a single source of truth for runtime parameters

## Key Files
- `assistant.yaml` – main configuration file
- `config.py` – config loader and accessor logic

This layer should remain *dumb*: no business logic, only structured data.
