
# Tools

Low-level implementations used by built-in integrations.

## Responsibilities
- Communicating with concrete backends such as SearXNG
- Executing shell commands under the local approval policy
- Keeping backend-specific retries and result formatting out of the registry

`search_summarizer.py` converts concrete SearXNG results into bounded text returned by
the web-search tool. Model-facing capability schemas remain under `app/integrations/`.

Model-facing names, schemas, validation, and typed results live in `app/integrations`.
