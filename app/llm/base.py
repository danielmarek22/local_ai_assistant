from abc import ABC, abstractmethod
from typing import Iterator, List, Dict


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict]) -> str:
        """
        Blocking, non-streaming call.
        Must return the full assistant message.
        Used for planners, summarizers, classifiers, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def stream_chat(self, messages: List[Dict]) -> Iterator[str]:
        """
        Streaming call.
        Yields text chunks for user-facing responses.
        """
        raise NotImplementedError
