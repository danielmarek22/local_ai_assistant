from abc import ABC, abstractmethod
from typing import Iterator, List, Dict


class InferenceFailure(RuntimeError):
    """Typed model inference failure safe for orchestrator recovery decisions."""

    def __init__(self, category: str, message: str, *, cancelled: bool = False):
        super().__init__(message)
        self.category = category
        self.cancelled = cancelled


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        think_override=None,
        options_override: Dict | None = None,
        format_override: Dict | str | None = None,
    ) -> str:
        """
        Blocking, non-streaming call.
        Must return the full assistant message.
        `options_override` allows a caller to supply per-request generation
        options without mutating the client's default configuration.
        `format_override` optionally supplies a native structured-output format
        for blocking callers without affecting normal or streaming requests.
        Used for planners, summarizers, classifiers, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def stream_chat(self, messages: List[Dict], think_override=None) -> Iterator[str]:
        """
        Streaming call.
        Yields text chunks for user-facing responses.
        """
        raise NotImplementedError
