from dataclasses import dataclass
from typing import Any, Dict, Optional
import threading
import time

from app.perception.attachments import Attachment, ImageAttachment
from app.perception.keys import PerceptionKey

@dataclass(frozen=True)
class PerceptionEntry:
    value: Any
    timestamp: float

    @property
    def age(self) -> float:
        return time.time() - self.timestamp


class PerceptionState:
    """
    Shared, continuously updated world model.
    Written by perception producers.
    Read by planner / orchestrator.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: Dict[str, PerceptionEntry] = {}

    def update(self, key: PerceptionKey, value: Any) -> None:
        """Update or insert a perception signal."""
        with self._lock:
            self._entries[key.value] = PerceptionEntry(
                value=value,
                timestamp=time.time(),
            )

    def get(self, key: PerceptionKey) -> Optional[PerceptionEntry]:
        """Read a single perception entry."""
        with self._lock:
            return self._entries.get(key.value)

    def snapshot(self) -> Dict[str, Any]:
        """
        Planner-safe snapshot.
        Returns plain values so planner code does not depend on PerceptionEntry.
        """
        with self._lock:
            return {
                key: entry.value
                for key, entry in self._entries.items()
            }
