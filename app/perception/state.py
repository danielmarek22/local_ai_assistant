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
