
# Tools

Low-level implementations used by built-in integrations.

## Responsibilities
- Communicating with concrete backends such as SearXNG
- Executing shell commands under the local approval policy
- Keeping backend-specific retries and result formatting out of the registry

Model-facing names, schemas, validation, and typed results live in `app/integrations`.
