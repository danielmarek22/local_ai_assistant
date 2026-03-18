
# Storage

Persistent storage abstraction.

## Responsibilities
- Create and configure the SQLite connection
- Initialize and migrate core schema tables
- Expose the shared `Database` object used by store classes
- Expose the local vector database used for semantic retrieval

## Current Communication Pattern

- `app/storage/database.py` owns connection lifecycle and schema setup.
- `app/storage/vector_store.py` owns the persistent Chroma client and embedding-backed collections.
- `app/memory/chat_history.py`, `app/memory/memory_store.py`, and
  `app/memory/summary_store.py` use `Database.conn` directly to run SQL.
- `app/memory/chat_history.py` and `app/memory/memory_store.py` also write to Chroma collections.
- Higher-level components (orchestrator, services, planners) should call these
  stores rather than executing SQL directly.

## Current Backends

- SQLite
  - `chat_history`
  - `memory`
  - `conversation_summary`
- ChromaDB
  - `semantic_memory`
  - `episodic_memory`

This is a pragmatic local-first setup; a stricter repository interface can be
introduced later if backend portability becomes a requirement.
