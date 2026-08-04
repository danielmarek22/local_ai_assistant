from copy import deepcopy
from pathlib import Path
import yaml


class Config:
    def __init__(self, path: str = "./app/config/assistant.yaml"):
        with open(Path(path), "r") as f:
            self.raw = yaml.safe_load(f) or {}

        # Core sections
        self.llm = self.raw.get("llm", {})
        self.assistant = self.raw.get("assistant", {})

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
        self.context = self._load_context_config(self.raw.get("context"))

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
        }

    def _load_context_config(self, raw_context: dict | None) -> dict:
        config = deepcopy(self._default_context_config())
        if not isinstance(raw_context, dict) or not raw_context:
            return config

        config.update(raw_context)
        return config
