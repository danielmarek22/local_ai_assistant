
# Memory

Memory management, retrieval, and summarization policy.

## Responsibilities
- Storing conversational and semantic memories
- Applying lightweight memory-write policy decisions
- Exposing retrieval APIs to the orchestrator
- Managing rolling conversation summaries

Memory should be treated as an *active system*, not just a database.

## Current Memory Layout

- `ChatHistoryStore`
  - stores every message in SQLite
  - stores an embedded copy in Chroma's `episodic_memory`
  - retrieves recent turns from SQLite
  - retrieves semantically similar past-session messages from Chroma
- `MemoryStore`
  - stores long-term fact-like memory in SQLite
  - stores embeddings in Chroma's `semantic_memory`
  - retrieves relevant facts semantically from Chroma
- `SummaryStore`
  - stores one summary per conversation session in SQLite
- `SimpleMemoryPolicy`
  - converts planner `write_memory` actions into concrete storage decisions

## How Memory Is Used In A Turn

1. The orchestrator receives user input.
2. Semantic memory is queried for relevant facts.
3. Episodic memory is queried for similar messages from past sessions.
4. Retrieved memory is injected into perception for planning and into prompt context for response generation.
5. The current user/assistant messages are persisted to both chat history layers.
6. If enough turns have accumulated and no summary exists yet, the session is summarized.

## Data Access

Memory store classes execute SQL through the shared SQLite connection provided
by `app/storage/database.py` and use `app/storage/vector_store.py` for Chroma
collections. This is a pragmatic hybrid design rather than a strict repository
layer.
