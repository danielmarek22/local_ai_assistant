from pathlib import Path
import yaml


class Config:
    def __init__(self, path: str = "./app/config/assistant.yaml"):
        with open(Path(path), "r") as f:
            self.raw = yaml.safe_load(f)

        self.llm = self.raw["llm"]
        self.assistant = self.raw["assistant"]
