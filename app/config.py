from pathlib import Path
import yaml


class Config:
    def __init__(self, path: str = "./app/config/assistant.yaml"):
        with open(Path(path), "r") as f:
            self.raw = yaml.safe_load(f) or {}

        # Core sections
        self.llm = self.raw.get("llm", {})
        self.assistant = self.raw.get("assistant", {})
        self.user_context = self.raw.get("user_context", {})

        # Planner
        self.planner = self.raw.get(
            "planner",
            {
                "mode": "rule",
                "llm_enabled": False,
                "timeout_ms": 4000,
            },
        )

        # Tools
        self.tools = self.raw.get("tools", {})

        # Orchestrator
        self.orchestrator = self.raw.get(
            "orchestrator",
            {
                "summary_trigger": 10,
            },
        )

        # Context
        self.context = self.raw.get(
            "context",
            {
                "history_limit": 6,
                "memory_limit": 5,
            },
        )

        # TTS
        self.tts = self.raw.get(
            "tts",
            {
                "model_path": "models/piper/en_US-amy-medium.onnx",
                "use_cuda": False,
            },
        )

        # Logging
        self.logging = self.raw.get(
            "logging",
            {
                "level": "INFO",
                "dir": "logs",
            },
        )
