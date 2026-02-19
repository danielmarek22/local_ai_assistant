
# Storage

Persistent storage abstraction.

## Responsibilities
- Create and configure the SQLite connection
- Initialize and migrate core schema tables
- Expose the shared `Database` object used by store classes

## Current Communication Pattern

- `app/storage/database.py` owns connection lifecycle and schema setup.
- `app/memory/chat_history.py`, `app/memory/memory_store.py`, and
  `app/memory/summary_store.py` use `Database.conn` directly to run SQL.
- Higher-level components (orchestrator, services, planners) should call these
  stores rather than executing SQL directly.

This is a pragmatic local-first setup; a stricter repository interface can be
introduced later if backend portability becomes a requirement.
