"""Attachment files only; SQL transactions and deletion fences belong to the caller."""

import hashlib
import mimetypes
import shutil
from pathlib import Path

from app.core.session_ids import validate_session_id
from app.logging import trace_event
from app.paths import STATIC_DIR, resolve_app_path
from app.perception.attachments import Attachment, ImageAttachment


class AttachmentFiles:
    def __init__(self, uploads_root: str):
        self.uploads_root = resolve_app_path(uploads_root)
        self.uploads_root.mkdir(parents=True, exist_ok=True)

    def store(self, session_id: str, message_id: int, attachment: Attachment) -> ImageAttachment:
        message_dir = self.session_dir(session_id) / str(message_id)
        message_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(attachment, ImageAttachment):
            raise ValueError(
                f"Unsupported attachment type for chat history storage: {attachment.__class__.__name__}"
            )

        payload = attachment.as_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        file_path = message_dir / f"{sha256}{self.extension_for_mime_type(attachment.mime_type)}"
        if not file_path.exists():
            file_path.write_bytes(payload)
        trace_event(
            "chat_history",
            "attachment_stored",
            session_id=session_id,
            payload={
                "file_path": str(file_path),
                "bytes": len(payload),
                "name": attachment.name,
            },
        )

        return ImageAttachment(
            name=attachment.name,
            mime_type=attachment.mime_type,
            size_bytes=attachment.size_bytes,
            storage_path=str(file_path),
            sha256=sha256,
            summary_text=(
                attachment.summary_text.strip()
                if isinstance(attachment.summary_text, str)
                and attachment.summary_text.strip()
                else None
            ),
        )

    def delete_session(self, session_id: str) -> None:
        upload_dir = self.session_dir(session_id)
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

    def extension_for_mime_type(self, mime_type: str) -> str:
        extension = mimetypes.guess_extension(mime_type, strict=False) or ""
        if extension == ".jpe":
            return ".jpg"
        if extension:
            return extension
        return ".img"

    def public_url(self, storage_path: str) -> str:
        path = Path(storage_path)
        try:
            relative_path = (
                path.relative_to(STATIC_DIR)
                if path.is_absolute()
                else path.relative_to("static")
            )
        except ValueError:
            relative_path = path
        return f"/static/{relative_path.as_posix()}"

    def session_dir(self, session_id: str) -> Path:
        session_id = validate_session_id(session_id)
        upload_root = self.uploads_root.resolve()
        session_dir = (upload_root / session_id).resolve()
        if session_dir.parent != upload_root:
            raise ValueError("Invalid session ID: attachment path escapes upload root")
        return session_dir

