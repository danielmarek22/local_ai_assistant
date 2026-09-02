# Knowledge Inspector

The Knowledge inspector is a read-only HTTP and browser view over Astra's beliefs.

- **Effective state** uses the production snapshot service, including expiry,
  evidence-track specificity, ordering, and record limits.
- **All records** is an owner-scoped, paginated view that also includes expired and
  invalidated rows. Record status is derived at inspection time without changing data.
- **Context preview** uses the same provider, formatter, and section-rendering helper
  as production context construction.
- **Saved memories** lists every global structured record from SQLite's canonical
  `memory` table through an inspection-only read. It exposes the persisted ID,
  category, content, importance scalar, creation time, and last-accessed time.

Inspection is independent of conversational extraction. Disabling extraction stops
new conversational belief production but does not hide stored beliefs or prevent their
normal snapshot/context use. The inspector never edits beliefs, exposes generic SQL,
or changes access timestamps.

Saved-memory inspection never calls semantic retrieval, Chroma, embeddings, models,
or memory tools, and therefore never updates `last_accessed_at`. These records are
global and survive individual chat-session deletion. The current schema does not store
source session or message provenance, so the UI states that limitation explicitly.

The application currently assumes a trusted localhost deployment and has no HTTP
authentication layer. Summary inspection, vector search, and knowledge-management
actions remain deferred.
