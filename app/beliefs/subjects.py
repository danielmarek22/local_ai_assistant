from __future__ import annotations

import re
import unicodedata

from app.beliefs.models import AllowedSubject, SubjectKind


_FIRST_PERSON_REFERENCES = {"i", "i m", "i ve", "i d", "my", "me", "myself"}
_UNRESOLVED_REFERENCES = {
    "he", "her", "hers", "him", "his", "it", "its", "she", "their", "theirs",
    "them", "they", "you", "your", "yours",
}


WORLD_SUBJECT = AllowedSubject(
    "entity:world",
    SubjectKind.WORLD,
    "World",
    ("World",),
    "Application-owned subject for explicitly stated world state.",
)
DEFAULT_ENVIRONMENT_SUBJECT = AllowedSubject(
    "entity:environment:default",
    SubjectKind.ENVIRONMENT,
    "Environment",
    ("Environment",),
    "Application-owned subject for explicitly stated environment state.",
)


def participant_subject(sender_id: str, display_name: str, sender_type: str) -> AllowedSubject:
    if not sender_id or len(sender_id) > 128:
        raise ValueError("Authoritative sender ID must contain 1-128 characters")
    kind = SubjectKind.AGENT if sender_type == "external_agent" else SubjectKind.PERSON
    return AllowedSubject(
        sender_id,
        kind,
        display_name,
        (display_name,),
        "Authoritatively attributed conversation participant.",
    )


def default_allowed_subjects(sender_id: str, display_name: str, sender_type: str):
    return [
        participant_subject(sender_id, display_name, sender_type),
        WORLD_SUBJECT,
        DEFAULT_ENVIRONMENT_SUBJECT,
    ]


def subject_from_belief(belief) -> AllowedSubject:
    return AllowedSubject(
        belief.subject_id,
        belief.subject_kind,
        belief.subject_display_name,
    )


def resolve_subject_reference(
    subject_reference: str | None,
    user_text: str,
    allowed_subjects: list[AllowedSubject],
    source_sender_id: str,
) -> AllowedSubject:
    """Resolve a textual referent to exactly one application-owned subject."""
    if not subject_reference:
        raise ValueError("Conversational ASSERT requires subject_reference")
    if subject_reference not in user_text:
        raise ValueError("Belief subject_reference is not an exact source-message substring")

    normalized_reference = _normalize_reference(subject_reference)
    if not normalized_reference:
        raise ValueError("Belief subject_reference is empty after normalization")
    allowed_by_id = {subject.subject_id: subject for subject in allowed_subjects}

    if normalized_reference in _FIRST_PERSON_REFERENCES:
        source = allowed_by_id.get(source_sender_id)
        if source is None or source.subject_kind not in {SubjectKind.PERSON, SubjectKind.AGENT}:
            raise ValueError("Authoritative source sender is not an allowed participant subject")
        return source
    if normalized_reference in _UNRESOLVED_REFERENCES:
        raise ValueError("Belief subject_reference is unresolved or second-person")

    matches = {
        subject.subject_id: subject
        for subject in allowed_subjects
        if any(
            normalized_reference == _normalize_reference(label)
            and _normalized_label_occurs(user_text, label)
            for label in _subject_labels(subject)
        )
    }
    if len(matches) != 1:
        raise ValueError("Belief subject_reference is unknown or ambiguous")
    return next(iter(matches.values()))


def _subject_labels(subject: AllowedSubject) -> tuple[str, ...]:
    return subject.subject_reference_labels or (subject.subject_display_name,)


def _normalize_reference(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(part for part in re.split(r"[^\w]+", normalized) if part)


def _normalized_label_occurs(text: str, label: str) -> bool:
    normalized_text = f" {_normalize_reference(text)} "
    normalized_label = _normalize_reference(label)
    return bool(normalized_label) and f" {normalized_label} " in normalized_text
