
# Storage

Persistent storage abstraction.

## Responsibilities
- Create and configure the SQLite connection
- Initialize and migrate core schema tables
- Expose the shared `Database` object used by store classes
- Expose the local vector database used for semantic retrieval
- Support persisted chat attachments and their retrieval metadata

## Current Communication Pattern

- `app/storage/database.py` owns connection lifecycle and schema setup.
- `app/storage/vector_store.py` owns the persistent Chroma client and embedding-backed collections.
- `app/memory/chat_history.py`, `app/memory/memory_store.py`, and
  `app/memory/summary_store.py` use `Database.conn` directly to run SQL.
- `app/memory/chat_history.py` and `app/memory/memory_store.py` also write to Chroma collections.
- Chat attachments are stored as files on disk, while their metadata lives in SQLite.
- Higher-level components (orchestrator, services, planners) should call these
  stores rather than executing SQL directly.
- `chat_sessions` stores the authoritative `direct` or `manual_group` kind.
- `chat_history` keeps model role separate from `sender_id`,
  `sender_display_name`, `sender_type`, and `input_source`. Legacy nullable
  sender fields are resolved deterministically when rows are read.

## Current Backends

- SQLite
  - `chat_history`
  - `chat_attachments`
  - `memory`
  - `conversation_summary`
- ChromaDB
  - `semantic_memory`
  - `episodic_memory`
    - conversation turns
    - image attachment summaries for long-term retrieval

This is a pragmatic local-first setup; a stricter repository interface can be
introduced later if backend portability becomes a requirement.
