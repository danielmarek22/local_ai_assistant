from copy import deepcopy
import logging
from pathlib import Path
import yaml

from app.paths import DEFAULT_CONFIG_PATH


logger = logging.getLogger("config")


class Config:
    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH):
        self.path = Path(path).expanduser().resolve()
        with self.path.open("r") as f:
            self.raw = yaml.safe_load(f) or {}

        # Core sections
        self.llm = self.raw.get("llm", {})
        self.assistant = self.raw.get("assistant", {})
        self.local_human = self._load_local_human_config(self.raw.get("local_human"))

        # Planner
        self.planner = self.raw.get(
            "planner",
            {
                "mode": "rule",
                "llm_enabled": False,
                "timeout_ms": 4000,
            },
        )

        # Integrations and temporary legacy tool configuration
        self.tools = self.raw.get("tools", {})
        self.integrations = self._load_integrations_config(
            self.raw.get("integrations"),
            self.tools,
        )

        # Orchestrator
        self.orchestrator = self.raw.get(
            "orchestrator",
            {
                "summary_trigger": 10,
            },
        )

        # Context
        self.context = self._load_context_config(self.raw.get("context"))

        self.beliefs = self._load_beliefs_config(self.raw.get("beliefs"))

        self.autonomy = self._load_autonomy_config(self.raw.get("autonomy"))

        # TTS
        self.tts = self._load_tts_config(self.raw.get("tts"))

        # STT
        self.stt = self.raw.get(
            "stt",
            {
                "enabled": True,
                "model_size": "small",
                "device": "cpu",
                "compute_type": "int8",
                "vad_filter": True,
                "vad_parameters": {"min_silence_duration_ms": 300},
            },
        )

        # Voice input routing
        self.voice_input = self._load_voice_input_config(self.raw.get("voice_input"))

        # Logging
        self.logging = self.raw.get(
            "logging",
            {
                "level": "INFO",
                "console_level": "INFO",
                "file_level": "INFO",
                "dir": "logs",
                "file_name": "assistant.log",
                "max_bytes": 10_000_000,
                "backup_count": 5,
                "trace_enabled": True,
                "trace_level": "DEBUG",
                "trace_file_name": "trace.log",
                "trace_max_bytes": 10_000_000,
                "trace_backup_count": 5,
            },
        )

    @staticmethod
    def _load_local_human_config(raw_local_human: dict | None) -> dict:
        config = {"id": "local-human", "display_name": "You"}
        if isinstance(raw_local_human, dict):
            config.update(raw_local_human)
        config["id"] = str(config.get("id") or "local-human").strip() or "local-human"
        config["display_name"] = str(config.get("display_name") or "You").strip() or "You"
        return config

    @staticmethod
    def _default_tts_config() -> dict:
        return {
            "engine": "qwen3",
            "gpt_sovits": {
                "api_url": "http://127.0.0.1:9880/tts",
                "ref_audio_path": "",
                "prompt_text": "",
                "text_lang": "en",
                "prompt_lang": "en",
            },
            "qwen3": {
                "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "device": "cuda:0",
                "speaker": "Ryan",
                "language": "English",
                "ref_audio_path": "app/tts/sample.wav",
                "ref_text": (
                    "Early in my career, I worked at a small restaurant in the "
                    "Vasari Passage. One day, entirely out of the blue, Lady "
                    "Furina decided to dine at our restaurant after a performance. "
                    "As I was scrambling around the back, a customer asked to meet "
                    "the head chef. So, I rushed out to their table, knife still in "
                    "hand, and completely froze. Because there I was, standing "
                    "face-to-face with Lady Furina of all people. I knew I had to "
                    "make something special, and it was like my instincts took over. "
                    "I came up with a Lily sugar-glazed opera cake, and Lady Furina "
                    "actually liked it and said she would remember me. I barely slept "
                    "a wink that night, and the very next day, there was a position "
                    "waiting for me at the Palais Mermonia."
                ),
            },
            "piper": {
                "model_path": "models/piper/en_US-amy-medium.onnx",
                "use_cuda": False,
            },
        }

    def _load_tts_config(self, raw_tts: dict | None) -> dict:
        config = deepcopy(self._default_tts_config())
        if not isinstance(raw_tts, dict) or not raw_tts:
            return config

        engine = raw_tts.get("engine") or raw_tts.get("provider") or raw_tts.get("backend")

        if any(
            key in raw_tts
            for key in ("gpt_sovits", "qwen3", "piper", "engine", "provider", "backend")
        ):
            if engine:
                config["engine"] = str(engine)

            for key in ("gpt_sovits", "qwen3", "piper"):
                engine_config = raw_tts.get(key)
                if isinstance(engine_config, dict):
                    config[key].update(engine_config)
            return config

        if any(
            key in raw_tts
            for key in ("api_url", "ref_audio_path", "prompt_text", "text_lang", "prompt_lang")
        ):
            config["engine"] = str(engine or "gpt_sovits")
            config["gpt_sovits"].update(raw_tts)
            return config

        if any(key in raw_tts for key in ("model_path", "use_cuda")):
            config["engine"] = str(engine or "piper")
            config["piper"].update(raw_tts)
            return config

        config["engine"] = str(engine or "qwen3")
        config["qwen3"].update(raw_tts)
        return config

    @staticmethod
    def _default_voice_input_config() -> dict:
        return {
            "path": "stt",
            "native_audio": {
                "payload_field": "images",
                "prompt_text": "Please answer the user's spoken audio.",
                "display_text": "Voice message",
                "convert_to_wav": True,
                "sample_rate": 16000,
            },
        }

    def _load_voice_input_config(self, raw_voice_input: dict | None) -> dict:
        config = deepcopy(self._default_voice_input_config())
        if not isinstance(raw_voice_input, dict) or not raw_voice_input:
            return config

        path = raw_voice_input.get("path")
        if path:
            config["path"] = str(path)

        native_audio = raw_voice_input.get("native_audio")
        if isinstance(native_audio, dict):
            config["native_audio"].update(native_audio)

        return config

    @staticmethod
    def _default_context_config() -> dict:
        return {
            "history_limit": 6,
            "injected_memory_limit": 5,
            "integration_context_limit": 4000,
        }

    def _load_context_config(self, raw_context: dict | None) -> dict:
        config = deepcopy(self._default_context_config())
        if not isinstance(raw_context, dict) or not raw_context:
            return config

        config.update(raw_context)
        return config

    @staticmethod
    def _load_beliefs_config(raw_beliefs: dict | None) -> dict:
        config = {
            "enabled": True,
            "processing_mode": "disabled",
            "timezone": "UTC",
            "max_candidates": 4,
            "max_existing_beliefs": 24,
            "max_snapshot_chars": 2000,
            "max_disambiguating_context_chars": 1000,
            "max_generation_tokens": 384,
            "timeout_s": 30.0,
            "max_expiry_days": 90,
        }
        if isinstance(raw_beliefs, dict):
            if "extraction_enabled" in raw_beliefs:
                raise ValueError(
                    "beliefs.extraction_enabled was replaced by beliefs.processing_mode; "
                    "valid values are: disabled, observer, react_tool"
                )
            config.update(raw_beliefs)
        valid_modes = {"disabled", "observer", "react_tool"}
        mode = config.get("processing_mode")
        if mode not in valid_modes:
            raise ValueError(
                "beliefs.processing_mode must be one of: disabled, observer, react_tool"
            )
        if not bool(config.get("enabled", True)) and mode != "disabled":
            raise ValueError(
                "beliefs.enabled=false requires beliefs.processing_mode=disabled"
            )
        return config

    @staticmethod
    def _load_autonomy_config(raw_autonomy: dict | None) -> dict:
        config = {
            "enabled": False,
            "max_chain_events": 20,
            "max_chain_age_s": 1800,
            "max_queue_size": 256,
            "max_tool_steps": 5,
            "global_llm_concurrency": 1,
            "approval_timeout_s": 300,
            "recent_context_limit": 4000,
        }
        if isinstance(raw_autonomy, dict):
            config.update(raw_autonomy)
        return config

    @staticmethod
    def _load_integrations_config(
        raw_integrations: dict | None,
        legacy_tools: dict | None,
    ) -> dict:
        integrations = deepcopy(raw_integrations) if isinstance(raw_integrations, dict) else {}
        integrations.setdefault("memory", {"enabled": True})
        integrations.setdefault("shell", {"enabled": True, "timeout": 15})
        integrations.setdefault("mindcraft", {
            "enabled": False,
            "url": "http://localhost:8081",
            "agent_name": "",
            "connect_timeout": 3.0,
            "reconnect_delay_s": 2.0,
            "reconnect_max_delay_s": 30.0,
            "context_enabled": True,
            "recent_output_limit": 3,
            "events_enabled": True,
            "ambient_session_id": "",
        })

        legacy_web = legacy_tools.get("web") if isinstance(legacy_tools, dict) else None
        if "web" not in integrations and isinstance(legacy_web, dict):
            logger.warning(
                "Configuration key 'tools.web' is deprecated; use 'integrations.web'"
            )
            integrations["web"] = deepcopy(legacy_web)

        for name, integration_config in integrations.items():
            if not isinstance(name, str) or not isinstance(integration_config, dict):
                raise ValueError(f"Invalid integration configuration: {name!r}")

        return integrations
