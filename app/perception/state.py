import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import threading
import time


@dataclass(frozen=True)
class PerceptionEntry:
    value: Any
    timestamp: float

    @property
    def age(self) -> float:
        return time.time() - self.timestamp


@dataclass(frozen=True)
class ImageAttachment:
    name: str
    mime_type: str
    base64_data: str | None = None
    size_bytes: int | None = None
    attachment_id: int | None = None
    storage_path: str | None = None
    url: str | None = None
    sha256: str | None = None
    summary_text: str | None = None

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
            base64_data=None,
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

    def to_llm_image(self) -> str:
        if self.base64_data:
            return self.base64_data

        return base64.b64encode(self.as_bytes()).decode("ascii")

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


class PerceptionState:
    """
    Shared, continuously updated world model.
    Written by perception producers.
    Read by planner / orchestrator.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: Dict[str, PerceptionEntry] = {}

    def update(self, key: str, value: Any) -> None:
        """Update or insert a perception signal."""
        with self._lock:
            self._entries[key] = PerceptionEntry(
                value=value,
                timestamp=time.time(),
            )

    def get(self, key: str) -> Optional[PerceptionEntry]:
        """Read a single perception entry."""
        with self._lock:
            return self._entries.get(key)

    def snapshot(self) -> Dict[str, PerceptionEntry]:
        """
        Planner-safe snapshot.
        Returns shallow copy so planner can't mutate state.
        """
        with self._lock:
            return dict(self._entries)
