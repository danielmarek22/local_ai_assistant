from dataclasses import dataclass, field

from app.perception.state import ImageAttachment


@dataclass
class TurnInput:
    user_text: str
    attachments: list[ImageAttachment] = field(default_factory=list)
    think_override: bool | None = None

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
        if names:
            return f"user shared image attachments: {names}"
        return "user shared image attachments"

    def history_text(self) -> str:
        text = self.user_text.strip()
        if not self.attachments:
            return text

        suffix = f"User attached {len(self.attachments)} image"
        if len(self.attachments) != 1:
            suffix += "s"

        if text:
            return f"{text}\n\n[{suffix}]"

        return f"[{suffix}]"
