from __future__ import annotations

from app.beliefs.models import AllowedSubject, SubjectKind


WORLD_SUBJECT = AllowedSubject("entity:world", SubjectKind.WORLD, "World")
DEFAULT_ENVIRONMENT_SUBJECT = AllowedSubject(
    "entity:environment:default", SubjectKind.ENVIRONMENT, "Environment"
)


def participant_subject(sender_id: str, display_name: str, sender_type: str) -> AllowedSubject:
    if not sender_id or len(sender_id) > 128:
        raise ValueError("Authoritative sender ID must contain 1-128 characters")
    kind = SubjectKind.AGENT if sender_type == "external_agent" else SubjectKind.PERSON
    return AllowedSubject(sender_id, kind, display_name)


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
