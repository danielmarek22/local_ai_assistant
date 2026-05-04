from dataclasses import dataclass, field
from enum import Enum

from app.perception.attachments import Attachment

class InputModality(str, Enum):
    TEXT = "text"
    VOICE = "voice"


@dataclass
class TurnInput:
    user_text: str
    attachments: list[Attachment] = field(default_factory=list)
    think_override: bool | None = None
    input_modality: InputModality = InputModality.TEXT

    def _attachment_noun(self) -> str:
        types = {
            attachment.mime_type.split("/", 1)[0]
            for attachment in self.attachments
            if attachment.mime_type
        }
        if types == {"image"}:
            return "image"
        return "file"

    def retrieval_text(self) -> str:
        text = self.user_text.strip()
        if text:
            return text

        if not self.attachments:
            return ""

        names = ", ".join(
            attachment.name
            for attachment in self.attachments[:3]
            if attachment.name
        )
        noun = self._attachment_noun()
        if names:
            return f"user shared {noun} attachments: {names}"
        return f"user shared {noun} attachments"

    def history_text(self) -> str:
        text = self.user_text.strip()
        if not self.attachments:
            return text

        suffix = f"User attached {len(self.attachments)} {self._attachment_noun()}"
        if len(self.attachments) != 1:
            suffix += "s"

        if text:
            return f"{text}\n\n[{suffix}]"

        return f"[{suffix}]"
