from pathlib import Path
import yaml


class Config:
    def __init__(self, path: str = "./app/config/assistant.yaml"):
        with open(Path(path), "r") as f:
            self.raw = yaml.safe_load(f)

        # Core sections
        self.llm = self.raw.get("llm", {})
        self.assistant = self.raw.get("assistant", {})

        # Planner
        self.planner = self.raw.get(
            "planner",
            {
                "mode": "rule",
                "llm_enabled": False,
                "timeout_ms": 1500,
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
