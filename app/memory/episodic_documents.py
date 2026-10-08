"""Pure formatting for episodic index documents and canonical vector IDs.

Callers resolve sender attribution and session visibility before formatting.
This module does not read storage or update the index.
"""

from app.core.conversation import SenderAttribution, SessionKind
from app.perception.attachments import ImageAttachment



def message_vector_id(message_id: int) -> str:
    return f"message:{message_id}"


def attachment_vector_id(attachment_id: int | None) -> str:
    if attachment_id is None:
        raise ValueError("Stored attachment is missing its canonical ID")
    return f"attachment:{attachment_id}"


def build_message_document(
    role: str,
    content: str,
    sender: SenderAttribution,
    session_kind: SessionKind,
) -> str:
    if session_kind == SessionKind.DIRECT:
        return f"{role.upper()}: {content}"
    return f"{sender.sender_type.value.upper()} {sender.sender_display_name}: {content}"


def build_attachment_document(
    role: str,
    content: str,
    attachment: ImageAttachment,
    sender: SenderAttribution,
    session_kind: SessionKind = SessionKind.DIRECT,
) -> str:
    subject = role.upper()
    if session_kind == SessionKind.MANUAL_GROUP:
        subject = f"{sender.sender_type.value.upper()} {sender.sender_display_name}"
    parts = [
        f"{subject} shared image '{attachment.name}'.",
        f"Image summary: {attachment.summary_text}",
    ]
    if content and not (
        content.startswith("[User attached ") and content.endswith(" image]")
    ) and not (
        content.startswith("[User attached ") and content.endswith(" images]")
    ):
        parts.append(f"Related message text: {content}")
    return " ".join(parts)
