# Conversational Beliefs

Beliefs are key-value, revisable assertions owned by an agent. `owner_agent_id` is
the identity whose epistemic state is represented (Astra), while `subject_id` is the
entity the assertion concerns and `source_sender_id` is the evidence supplier. They are
not persona, memory, raw observations, goals, or integration state. Conversation
extraction uses the persisted participant message as canonical evidence.

Participant subject IDs reuse authoritative sender IDs. Application-owned subjects use
reserved IDs such as `entity:world` and `entity:environment:default`. Relay IDs are
derived from normalized display names in v1, so renaming a relay participant creates a
new identity; existing beliefs are not automatically reassigned.

Epistemic status is deterministic: a subject matching the source sender is a
`SELF_REPORT`; every other subject is an `ATTRIBUTED_CLAIM`. Different source tracks
coexist, and the system does not score trust or resolve conflicts. Each track retains
only its latest JSON value; lists may represent multi-valued properties.

Conversational extraction accepts only ordinary human or external-agent participant
turns. Assistant, system, tool, integration-runtime, and synthetic turns are excluded.
Affirmative extractor output is a complete assertion, never a model-selected create or
update target. Application code normalizes its predicate and atomically upserts the
logical track keyed by owner, visibility/session scope, subject, source sender, and
derived epistemic status. Explicit belief IDs are used only for authorized retraction.

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

Deleting a session removes only its `SESSION_CURRENT` beliefs. `AGENT_CURRENT`
beliefs remain available even when their latest source session or source message has
been deleted. Removing durable global beliefs requires a separate explicit forgetting
operation.
