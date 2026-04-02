import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Attachment:
    name: str
    mime_type: str
    size_bytes: int | None = None
    attachment_id: int | None = None
    storage_path: str | None = None
    url: str | None = None
    sha256: str | None = None
    summary_text: str | None = None

    def to_perception_payload(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }
        if self.attachment_id is not None:
            payload["id"] = self.attachment_id
        if self.url:
            payload["url"] = self.url
        if self.summary_text:
            payload["summary_text"] = self.summary_text
        return payload

    def to_api_payload(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }
        if self.attachment_id is not None:
            payload["id"] = self.attachment_id
        if self.url:
            payload["url"] = self.url
        if self.summary_text:
            payload["summary_text"] = self.summary_text
        return payload


@dataclass(frozen=True)
class ImageAttachment(Attachment):
    base64_data: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ImageAttachment":
        if not isinstance(payload, dict):
            raise ValueError("Image attachment must be an object")

        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Image attachment is missing a valid name")

        mime_type = payload.get("mime_type") or payload.get("mimeType")
        if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
            raise ValueError("Image attachment must include an image mime_type")

        raw_data = payload.get("data") or payload.get("base64_data")
        if not isinstance(raw_data, str) or not raw_data.strip():
            raise ValueError("Image attachment is missing base64 data")

        normalized_data = cls._normalize_base64(raw_data)

        size_bytes = payload.get("size_bytes")
        if size_bytes is None:
            size_bytes = payload.get("size")
        if size_bytes is not None and not isinstance(size_bytes, int):
            raise ValueError("Image attachment size_bytes must be an integer")

        return cls(
            name=name.strip(),
            mime_type=mime_type,
            base64_data=normalized_data,
            size_bytes=size_bytes,
        )

    @classmethod
    def from_stored_record(cls, payload: dict[str, Any]) -> "ImageAttachment":
        if not isinstance(payload, dict):
            raise ValueError("Stored image attachment must be an object")

        return cls(
            name=str(payload.get("name") or "image"),
            mime_type=str(payload.get("mime_type") or payload.get("mimeType") or "image/png"),
            size_bytes=payload.get("size_bytes"),
            attachment_id=payload.get("id"),
            storage_path=payload.get("storage_path"),
            url=payload.get("url"),
            sha256=payload.get("sha256"),
            summary_text=payload.get("summary_text"),
        )

    @staticmethod
    def _normalize_base64(value: str) -> str:
        normalized = value.strip()

        if normalized.startswith("data:") and "," in normalized:
            _, normalized = normalized.split(",", 1)

        try:
            base64.b64decode(normalized, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Image attachment data is not valid base64") from exc

        return normalized

    def as_bytes(self) -> bytes:
        if self.base64_data:
            return base64.b64decode(self.base64_data)

        if self.storage_path:
            return Path(self.storage_path).read_bytes()

        raise ValueError("Image attachment does not contain data or storage path")

    def to_llm_image(self) -> str:
        if self.base64_data:
            return self.base64_data

        return base64.b64encode(self.as_bytes()).decode("ascii")


def attachment_from_payload(payload: dict[str, Any]) -> Attachment:
    mime_type = payload.get("mime_type") or payload.get("mimeType")
    if isinstance(mime_type, str) and mime_type.startswith("image/"):
        return ImageAttachment.from_payload(payload)

    raise ValueError("Unsupported attachment mime_type")


def attachment_from_stored_record(payload: dict[str, Any]) -> Attachment:
    mime_type = payload.get("mime_type") or payload.get("mimeType")
    if isinstance(mime_type, str) and mime_type.startswith("image/"):
        return ImageAttachment.from_stored_record(payload)

    raise ValueError("Unsupported stored attachment mime_type")
