---
title: Notebook
description: Persistent free-form notes with explicit tools and automatic semantic recall.
---

# Notebook

Astra's scratchpad is a persistent collection of free-form notes. The model chooses what to write and can revise or remove individual entries. It coexists with the existing memory and belief systems; entries have no required categories, importance, or confidence scores.

## Tools

| Capability | Use |
| --- | --- |
| `scratchpad__write(content)` | Save a new note and return its ID and revision. |
| `scratchpad__read(id)` | Read the complete current note. |
| `scratchpad__recent(limit?, offset?)` | Browse recent notes with bounded previews and pagination. |
| `scratchpad__search(query, limit?)` | Search semantically and return relevant previews. |
| `scratchpad__edit(id, content, expected_revision)` | Replace content while retaining the ID. |
| `scratchpad__delete(id, expected_revision)` | Remove a note. |

An edit or deletion requires the current revision returned by read, search, recent, or automatic recall. If another operation has changed the note, read it again before deciding whether to retry. Notes can contain up to 4,000 characters; search queries can contain up to 2,000. Search defaults to five results, recent to ten, with a maximum of twenty. List results include `truncated`, `returned_count`, and `omitted_count`; recent also returns `next_offset`. Use read for full content.

## Automatic recall

Before the first inference of a conversational turn, including instant mode, a small provider searches for notes using the current message and up to two preceding conversational messages. The query reserves space for current text and omits tool/system records, failed messages, and the just-persisted current message. Attachment-only turns use a short attachment description. Retrieved notes are not fed back into the query.

Recall injects up to three relevant notes into a separate bounded section of the system context. Entries carry IDs, timestamps, revisions, and explicit truncation. Notes are revisable background data; their content does not override current instructions or authorize actions. Recall does not add recent notes merely because they are recent, and it does not run again on every tool-loop inference. No additional text-generation call is needed.

Autonomous events receive notebook recall only when their declared capability allowlist explicitly includes `scratchpad__read` or `scratchpad__search`. Proactive vision receives no notebook recall by default. Notebook tools use the existing dispatch authority checks; registering them does not grant them to events automatically.

## Storage and failures

SQLite's `scratchpad` table stores canonical content, scoped to the configured assistant ID. A separate Chroma collection supplies semantic candidates using the existing local CPU embedding model. Search checks the owner and revision against current SQLite rows and returns canonical text. Missing, deleted, foreign, stale, and below-threshold candidates are skipped.

The canonical mutation commits before indexing. If indexing fails, the tool still confirms the save or deletion with `indexed: false` and a repair warning. Do not repeat a successful save because its vector update failed. Read/recent and canonical mutations remain usable; explicit search reports an outage as unavailable, while passive recall logs it and continues the conversation.

One store serializes mutations and indexing within the running application. Embedding does not hold the shared database lock. Offline repair reindexes canonical notes and removes orphan vectors belonging to the current owner:

```bash
venv_app/bin/python main.py reconcile-indexes
```

Stop the application before this maintenance command. It rebuilds semantic memory, episodic history, and the enabled notebook; it does not coordinate concurrent application processes. The notebook repair affects the configured assistant's notes, preserving other owners' vectors.

The notebook belongs to the assistant across conversations, including local manual groups. Deleting a conversation leaves its independent notes intact, but a deleted originating conversation cannot perform a late note mutation. There is no automatic expiration, consolidation, or promotion into memory/beliefs.

## Configuration

```yaml
integrations:
  scratchpad:
    enabled: true
    recall_limit: 3
    max_distance: 0.70
    query_max_chars: 2000
    context_max_chars: 3000
```

These defaults also apply to existing configurations that omit the section. Disabling the feature removes tools and recall while preserving saved entries. Cosine distance must be between zero and two; smaller distances mean closer matches. The initial threshold is provisional and should be evaluated with ordinary use. Recall limit is 1–20, query budget 1–2,000 characters, and context payload budget 300–12,000 characters.

Recall duration, matched IDs, and failure status are recorded under `scratchpad_context.scratchpad_retrieval` in trace logs. Normal tool traces report note mutations; the context trace includes the exact notebook section supplied to the model.

Future Demo and Relay profiles must explicitly isolate storage or disable the notebook. Assistant ownership alone does not isolate profiles that share personal storage.

## Code boundaries

- `app/scratchpad/store.py`: canonical CRUD, revision checks, vector synchronization/search, and repair.
- `app/scratchpad/context.py`: bounded query construction and JSON preview formatting.
- `app/integrations/scratchpad.py`: validated native tools and result handling.
- `app/core/orchestrator_factory.py`: configuration and composition using shared resource owners.
- `app/core/orchestrator.py`: one recall operation per eligible turn.
- `app/core/context_builder.py`: dedicated prompt section without database access.
