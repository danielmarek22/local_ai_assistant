
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
  - stores attachment metadata in SQLite and image bytes on disk under `static/uploads/`
  - stores embedded message copies in Chroma's `episodic_memory`
  - stores one retrieval-oriented summary per saved image in `episodic_memory`
  - retrieves recent turns from SQLite
  - retrieves semantically similar past-session messages and stored image summaries from Chroma
- `MemoryStore`
  - stores long-term fact-like memory in SQLite
  - stores embeddings in Chroma's `semantic_memory`
  - retrieves relevant facts semantically from Chroma
  - exposes a separate SQLite-only, read-only inspection list for the Knowledge UI;
    this path does not query Chroma or update access timestamps
- `SummaryStore`
  - stores one summary per conversation session in SQLite
- `SimpleMemoryPolicy`
  - converts late-routed `write_memory` actions into concrete storage decisions

The attachment model now has a shared `Attachment` base type, but persistence is still intentionally image-focused today.

## How Memory Is Used In A Turn

1. The orchestrator receives user input.
2. Semantic memory is queried for relevant facts.
3. Episodic memory is queried for similar messages from past sessions.
4. Retrieved memory is injected into perception and into prompt context for response generation.
5. The current user/assistant messages are persisted to both chat history layers.
6. If the user attached images, those files are stored, summarized once for long-term retrieval, and linked back to the chat message.
7. Recent user turns with images can be replayed into the multimodal prompt from stored attachments.
8. If enough turns have accumulated and no summary exists yet, the session is summarized.

## Data Access

Memory store classes execute SQL through the shared SQLite connection provided
by `app/storage/database.py` and use `app/storage/vector_store.py` for Chroma
collections. This is a pragmatic hybrid design rather than a strict repository
layer.

## Deletion Semantics

Deleting a session removes:

- chat rows from SQLite
- attachment rows from SQLite
- uploaded image files on disk
- episodic vector entries for both text turns and image summaries

Long-term structured records in the global `memory` table are not session-owned and
therefore survive chat-session deletion. Their current schema has no source session or
message provenance.
