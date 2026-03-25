import base64
import binascii
from dataclasses import dataclass
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
    base64_data: str
    size_bytes: int | None = None

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

    def to_perception_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }

    def to_llm_image(self) -> str:
        return self.base64_data


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
