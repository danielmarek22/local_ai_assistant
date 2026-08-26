"""Small canonical predicate vocabulary for conversational current-state beliefs."""

import re


CANONICAL_PREDICATES: dict[str, str] = {
    "current_location": "The user's current physical location.",
    "current_activity": "What the user is currently doing.",
    "current_availability": "The user's current availability or interruptibility.",
    "temporary_physical_condition": "A temporary, explicitly stated physical condition.",
    "current_work_context": "The user's current work mode or work context.",
    "current_environment_status": "A revisable current state of the user's environment.",
    "current_conversation_context": "State explicitly limited to this conversation or session.",
    "preferred_beverage": "The participant's stable preferred beverage.",
}

# These aliases are normalization guardrails, not an ontology or inference system.
PREDICATE_ALIASES: dict[str, str] = {
    "location": "current_location",
    "current_place": "current_location",
    "activity": "current_activity",
    "current_task": "current_activity",
    "availability": "current_availability",
    "availability_status": "current_availability",
    "physical_condition": "temporary_physical_condition",
    "temporary_condition": "temporary_physical_condition",
    "work_context": "current_work_context",
    "environment_status": "current_environment_status",
    "conversation_context": "current_conversation_context",
    "beverage_preference": "preferred_beverage",
    "drink_preference": "preferred_beverage",
    "favorite_beverage": "preferred_beverage",
    "favorite_drink": "preferred_beverage",
    "preferred_drink": "preferred_beverage",
    "preferred_tea": "preferred_beverage",
    "preferred_coffee": "preferred_beverage",
}

_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def normalize_predicate(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    normalized = PREDICATE_ALIASES.get(normalized, normalized)
    if not _PREDICATE_RE.fullmatch(normalized):
        raise ValueError(
            "Belief predicates must use 2-64 lowercase snake_case characters"
        )
    return normalized


def validate_predicate_value_semantics(predicate: str, value) -> None:
    """Reject bounded, known value-bearing preference predicate shapes."""
    if isinstance(value, bool) and value and re.fullmatch(r"(?:prefers|likes)_[a-z0-9_]+", predicate):
        raise ValueError(
            "Preference assertions must use a stable property predicate and place the item in value"
        )
