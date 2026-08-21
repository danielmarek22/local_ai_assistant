# Conversational Beliefs

Beliefs are current, revisable assertions owned by an agent. They are not persona,
memory, raw observations, goals, or integration state. Conversation extraction uses the
persisted user message as canonical evidence and never extracts from assistant output.

Visibility is deliberately limited to `AGENT_CURRENT` (same owner across sessions) and
`SESSION_CURRENT` (originating session only). Application code supplies the owner and
session; the model can select only the visibility policy.

The starter predicate vocabulary is defined in `vocabulary.py`: `current_location`,
`current_activity`, `current_availability`, `temporary_physical_condition`,
`current_work_context`, `current_environment_status`, and
`current_conversation_context`. Extensions are allowed only when no starter predicate
fits and must use bounded lowercase snake_case. This is a normalization convention, not
a general ontology.

`beliefs.enabled` controls belief storage, lookup, deletion, and context injection.
Conversational LLM extraction is separately opt-in through
`beliefs.extraction_enabled`; it defaults to false.

Deleting a session removes its `SESSION_CURRENT` beliefs. It also removes an
`AGENT_CURRENT` belief when that session is the belief's latest provenance. An
owner-wide belief later updated from another session is retained.
