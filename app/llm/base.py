from abc import ABC, abstractmethod
from typing import Iterator

class LLMClient(ABC):
    @abstractmethod
    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """
        Streams tokens from the LLM.
        Yields text chunks.
        """
        pass
