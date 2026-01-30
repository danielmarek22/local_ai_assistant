from abc import ABC, abstractmethod
from pathlib import Path


class TTS(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: Path) -> None:
        pass
