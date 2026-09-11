
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
  - uses canonical message and attachment IDs for idempotent vector documents
  - can reconcile missing, stale, and orphaned episodic vector entries from SQLite
  - retrieves recent turns from SQLite
  - uses Chroma candidate IDs/scores to retrieve canonical past-session messages and image summaries from SQLite
- `MemoryStore`
  - stores long-term fact-like memory in SQLite
  - stores embeddings in Chroma's `semantic_memory`
  - uses the canonical SQLite UUID as the idempotent vector document ID
  - can reconcile missing, stale, and orphaned semantic vector entries from SQLite
  - uses Chroma candidate IDs/scores to retrieve current fact text from SQLite
  - exposes a separate SQLite-only, read-only inspection list for the Knowledge UI;
    this path does not query Chroma or update access timestamps
- `SummaryStore`
  - stores one summary per conversation session in SQLite
- `SimpleMemoryPolicy`
  - converts late-routed `write_memory` actions into concrete storage decisions
- `MemoryRetriever`
  - combines semantic and episodic retrieval for a turn
- `MemoryActionHandler`
  - applies policy-validated writes requested through the memory integration
- `MemoryReflector`
  - performs explicitly requested consolidation and pruning
- `HistorySummarizer`
  - creates rolling conversation summaries used by `TurnFinalizer`
  - receives the saved summary separately from new messages and treats both as data
  - rejects empty or malformed model output, preserving the saved summary and checkpoint

The attachment model now has a shared `Attachment` base type, but persistence is still intentionally image-focused today.

## How Memory Is Used In A Turn

1. The orchestrator receives user input.
2. The user message and supported attachments are saved before retrieval; retries reuse
   the existing message. A canonical persistence failure stops the turn.
3. Semantic and episodic memory are queried independently. Either provider may fail
   without suppressing results from the other or preventing response generation.
4. Available memory is injected into perception and prompt context. Failed provider
   names are recorded in retrieval results, logs, traces, and perception diagnostics;
   an outage is distinguished from a successful search with no matches.
5. The assistant response is persisted after generation. Canonical message writes
   survive vector-index write failures, which remain repairable through reconciliation.
6. If the user attached images, those files are stored, summarized once for long-term retrieval, and linked back to the chat message.
7. Current-turn images are sent as multimodal payloads. Historical image turns use their
   stored filename and summary as text, avoiding repeated binary payloads while keeping
   visual context retrievable.
8. If enough new messages have accumulated, create or update the session summary using
   its previous summary and the messages after its saved checkpoint. Failed or empty
   generations leave both the summary and checkpoint unchanged for a later attempt.

Summary checkpoints use the last summarized SQLite message ID, not a row count or
position in a recent-history window. Each completed turn can summarize one batch of
up to 100 eligible messages (or the configured trigger size if larger), in ascending
ID order. Only non-excluded user and assistant rows count toward the trigger; tool
records and other sessions do not consume the batch. Summary text and its checkpoint
are saved together only after successful generation. A backlog is drained gradually
as turns complete, rather than through an unbounded series of model calls.

Existing databases retain their summary text but start the new ID checkpoint at zero.
The legacy moving-window count cannot reliably identify covered messages, so migration
conservatively replays eligible history in batches into the saved summary. This may
revisit old facts, but avoids silently skipping them. Reopening the database preserves
new ID checkpoints; the old count column is retained only for schema compatibility.

## Data Access

Memory store classes execute SQL through the shared SQLite connection provided
by `app/storage/database.py` and use `app/storage/vector_store.py` for Chroma
collections. This is a pragmatic hybrid design rather than a strict repository
layer.

SQLite is authoritative for semantic memories. If a vector write or deletion fails,
`MemoryStore` raises `MemoryIndexSyncError` with the completed canonical change count
and affected IDs. `reconcile_index()` can then rebuild the semantic projection without
duplicating canonical rows.

Run `python main.py reconcile-indexes` with the application stopped to reconcile both
semantic and episodic Chroma collections from SQLite. The command initializes storage
only and prints a JSON report containing canonical, upserted, and removed counts.

Retrieval treats vector results as ranked candidate IDs, never authoritative text or
visibility metadata. Semantic candidates must still exist in SQLite, and their current
content is loaded before strict/fallback selection. Only selected canonical memories
have their access times updated. Episodic message and attachment IDs must resolve to
non-excluded history outside the current session and outside deleted sessions. Image
candidates also require a stored summary and a visible parent message; text and sender
labels are rebuilt from canonical records. Missing, unknown legacy, duplicate, and
invalid-score candidates are dropped without returning vector text as a fallback.

This keeps deleted or corrected content from being recalled from a stale index before
repair. Reconciliation is still needed for missing candidates, stale embeddings and
ranking quality. Retrieval does not repeatedly expand the search to replace rejected
candidates, so it can return fewer than the requested number while indexes diverge.

## Reflection

Manual reflection reads text from the blocking model client's message dictionary
using the same content validation as history summarization. Invalid or empty content,
invalid plans, and conflicting keep/delete decisions are rejected before mutation.
Deletion IDs are restricted to the reviewed stale set and deduplicated. If a deletion
source has disappeared during generation, consolidation aborts rather than partially
applying the plan.

Reflection deletes and replacement inserts commit together in one SQLite transaction.
Vector updates happen afterward; both replacement indexing and obsolete-vector cleanup
are attempted. A committed reflection returns `success: true` with
`index_sync_complete: false` and repair details if either index operation fails. The UI
distinguishes this from complete success. Use the reconciliation command described
above to repair the index; do not rerun reflection to recover an indexing failure.

## Deletion Semantics

Deleting a session removes:

- chat rows from SQLite
- attachment rows from SQLite
- uploaded image files on disk
- episodic vector entries for both text turns and image summaries
- the conversation summary and session-scoped beliefs

The deletion endpoint closes admission to new and queued turns, waits for active
work (including autonomous notification delivery), then removes the session data.
Deletion continues if the HTTP caller disconnects. A durable SQLite deletion marker
rejects late history, summary, belief, and integration writes, including after restart.
Queued integration events are discarded and pending operation records are cancelled;
the diagnostic journal remains. This does not undo effects already performed by an
external integration.

Existing browser connections close with code `4004`; the client reconnects with a
new conversation. Session replay buffers and outstanding approvals are cleared.
Deletion can wait for the current turn or delivery to finish; it does not interrupt
an in-flight model request or worker thread.

The deletion result distinguishes the canonical SQLite deletion from secondary cleanup.
Vector and filesystem cleanup are both attempted, and any failures are returned to the
API and shown as a warning in the history UI.

Agent-wide beliefs and long-term structured records in the global `memory` table are not session-owned and
therefore survive chat-session deletion. Their current schema has no source session or
message provenance.
