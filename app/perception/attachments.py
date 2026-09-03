import base64
import binascii
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_IMAGE_ATTACHMENT_BYTES = 10 * 1024 * 1024


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
        return self.to_perception_payload()


@dataclass(frozen=True)
class ImageAttachment(Attachment):
    base64_data: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        max_bytes: int = MAX_IMAGE_ATTACHMENT_BYTES,
    ) -> "ImageAttachment":
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

        normalized_data = cls._normalize_base64(raw_data, max_bytes=max_bytes)
        normalized_bytes = cls._decode_and_repair_image_bytes(
            normalized_data,
            mime_type,
            max_bytes=max_bytes,
        )

        size_bytes = payload.get("size_bytes")
        if size_bytes is None:
            size_bytes = payload.get("size")
        if size_bytes is not None and (
            type(size_bytes) is not int or size_bytes < 0
        ):
            raise ValueError("Image attachment size_bytes must be a non-negative integer")

        return cls(
            name=name.strip(),
            mime_type=mime_type,
            base64_data=base64.b64encode(normalized_bytes).decode("ascii"),
            size_bytes=len(normalized_bytes),
            sha256=hashlib.sha256(normalized_bytes).hexdigest(),
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
    def _normalize_base64(value: str, *, max_bytes: int) -> str:
        normalized = value.strip()

        if normalized.startswith("data:") and "," in normalized:
            _, normalized = normalized.split(",", 1)

        max_encoded_chars = 4 * ((max_bytes + 2) // 3)
        if len(normalized) > max_encoded_chars:
            raise ValueError(f"Image attachment exceeds the {max_bytes}-byte limit")

        try:
            base64.b64decode(normalized, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Image attachment data is not valid base64") from exc

        return normalized

    def as_bytes(self) -> bytes:
        if self.base64_data:
            decoded = base64.b64decode(self.base64_data)
            return self._repair_image_bytes(decoded, self.mime_type)

        if self.storage_path:
            decoded = Path(self.storage_path).read_bytes()
            return self._repair_image_bytes(decoded, self.mime_type)

        raise ValueError("Image attachment does not contain data or storage path")

    def to_llm_image(self) -> str:
        if self.base64_data:
            return self.base64_data

        return base64.b64encode(self.as_bytes()).decode("ascii")

    @classmethod
    def _decode_and_repair_image_bytes(
        cls,
        base64_value: str,
        mime_type: str,
        *,
        max_bytes: int,
    ) -> bytes:
        decoded = base64.b64decode(base64_value)
        if len(decoded) > max_bytes:
            raise ValueError(f"Image attachment exceeds the {max_bytes}-byte limit")
        return cls._repair_image_bytes(decoded, mime_type)

    @staticmethod
    def _repair_image_bytes(payload: bytes, mime_type: str) -> bytes:
        # Some clipboard providers prepend a few bytes before the real image
        # signature. Strip this prefix when we can confidently detect it.
        signature_map = {
            "image/png": b"\x89PNG\r\n\x1a\n",
            "image/jpeg": b"\xff\xd8\xff",
            "image/gif": b"GIF8",
        }
        signature = signature_map.get(mime_type)
        if signature is None or payload.startswith(signature):
            return payload

        max_prefix_scan = 32
        offset = payload.find(signature, 1, max_prefix_scan + len(signature))
        if offset <= 0:
            return payload

        return payload[offset:]


@dataclass(frozen=True)
class AudioAttachment(Attachment):
    base64_data: str | None = None

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        name: str = "voice.wav",
        mime_type: str = "audio/wav",
    ) -> "AudioAttachment":
        return cls(
            name=name,
            mime_type=mime_type,
            base64_data=base64.b64encode(payload).decode("ascii"),
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def as_bytes(self) -> bytes:
        if self.base64_data:
            return base64.b64decode(self.base64_data)

        if self.storage_path:
            return Path(self.storage_path).read_bytes()

        raise ValueError("Audio attachment does not contain data or storage path")

    def to_llm_audio(self) -> str:
        if self.base64_data:
            return self.base64_data

        return base64.b64encode(self.as_bytes()).decode("ascii")


def attachment_from_payload(
    payload: dict[str, Any],
    *,
    max_bytes: int = MAX_IMAGE_ATTACHMENT_BYTES,
) -> Attachment:
    if not isinstance(payload, dict):
        raise ValueError("Attachment must be an object")
    mime_type = payload.get("mime_type") or payload.get("mimeType")
    if isinstance(mime_type, str) and mime_type.startswith("image/"):
        return ImageAttachment.from_payload(payload, max_bytes=max_bytes)

    raise ValueError("Unsupported attachment mime_type")


def attachment_from_stored_record(payload: dict[str, Any]) -> Attachment:
    mime_type = payload.get("mime_type") or payload.get("mimeType")
    if isinstance(mime_type, str) and mime_type.startswith("image/"):
        return ImageAttachment.from_stored_record(payload)

    raise ValueError("Unsupported stored attachment mime_type")
