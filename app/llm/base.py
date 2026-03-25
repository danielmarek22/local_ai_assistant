from abc import ABC, abstractmethod
from typing import Iterator, List, Dict


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        think_override=None,
        options_override: Dict | None = None,
    ) -> str:
        """
        Blocking, non-streaming call.
        Must return the full assistant message.
        `options_override` allows a caller to supply per-request generation
        options without mutating the client's default configuration.
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
