from abc import ABC, abstractmethod
from typing import Iterator, List, Dict


def response_text(response: object, *, context: str = "Model") -> str:
    """Extract usable text from the message returned by a blocking chat call."""
    content = response.get("content") if isinstance(response, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{context} returned no usable text content")
    return content.strip()


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
    ) -> dict:
        """
        Blocking, non-streaming call.
        Must return the assistant message dictionary, with text in `content`.
        `options_override` allows a caller to supply per-request generation
        options without mutating the client's default configuration.
        `format_override` optionally supplies a native structured-output format
        for blocking callers without affecting normal or streaming requests.
        Used for summarizers, classifiers, and other structured workflows.
        """
        raise NotImplementedError

    @abstractmethod
    def stream_chat(self, messages: List[Dict], think_override=None) -> Iterator[str]:
        """
        Streaming call.
        Yields text chunks for user-facing responses.
        """
        raise NotImplementedError
