from copy import deepcopy
import logging
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
import yaml

from app.paths import DEFAULT_CONFIG_PATH


logger = logging.getLogger("config")

_CONFIG_SECTIONS = {
    "assistant",
    "autonomy",
    "beliefs",
    "context",
    "integrations",
    "llm",
    "local_human",
    "logging",
    "orchestrator",
    "stt",
    "tools",
    "tts",
    "vision_watchdog",
    "voice_input",
}


class _StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


_IDENTITY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _stripped_nonempty(value: str, *, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


class _LLMGenerationConfig(_StrictConfigModel):
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    num_predict: int | None = Field(default=None, gt=0)
    rep_pen: float | None = Field(default=None, gt=0.0)
    repeat_penalty: float | None = Field(default=None, gt=0.0)
    num_ctx: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_aliases(self):
        if self.max_tokens is not None and self.num_predict is not None:
            raise ValueError("max_tokens and num_predict are aliases; configure only one")
        if self.rep_pen is not None and self.repeat_penalty is not None:
            raise ValueError("rep_pen and repeat_penalty are aliases; configure only one")
        return self


class _LLMThinkingConfig(_StrictConfigModel):
    enabled: bool = False
    level: Literal["low", "medium", "high"] | None = None
    generation: _LLMGenerationConfig = Field(default_factory=_LLMGenerationConfig)


class _LLMConfig(_StrictConfigModel):
    backend: Literal["ollama"] = "ollama"
    host: str = "http://localhost:11434"
    model: str = "gemma4:12b-it-qat"
    timeout_s: float = Field(default=30.0, gt=0.0, le=3600.0)
    max_retries: int = Field(default=2, ge=0, le=20)
    retry_backoff_s: float = Field(default=0.25, ge=0.0, le=60.0)
    generation: _LLMGenerationConfig = Field(default_factory=_LLMGenerationConfig)
    thinking: _LLMThinkingConfig = Field(default_factory=_LLMThinkingConfig)

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        value = _stripped_nonempty(value, field_name="llm.host")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("llm.host must be an HTTP(S) origin without credentials or a path")
        return value.rstrip("/")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return _stripped_nonempty(value, field_name="llm.model")


class _IdentityConfig(_StrictConfigModel):
    id: str
    display_name: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if not _IDENTITY_ID_RE.fullmatch(value):
            raise ValueError(
                "must be 1-128 characters using letters, digits, '.', '_', ':', or '-', "
                "and must begin with a letter or digit"
            )
        return value

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        value = _stripped_nonempty(value, field_name="display_name")
        if len(value) > 128:
            raise ValueError("display_name must be at most 128 characters")
        return value


class _LocalHumanConfig(_IdentityConfig):
    id: str = "local-human"
    display_name: str = "You"


class _AssistantPersonalityConfig(_StrictConfigModel):
    default_emotion: str = "neutral"

    @field_validator("default_emotion")
    @classmethod
    def validate_default_emotion(cls, value: str) -> str:
        return _stripped_nonempty(value, field_name="assistant.personality.default_emotion")


class _AvatarControlsConfig(_StrictConfigModel):
    default_outfit: str = "default"
    expressions: list[str] | None = None

    @field_validator("default_outfit")
    @classmethod
    def validate_default_outfit(cls, value: str) -> str:
        return _stripped_nonempty(value, field_name="assistant.avatar_controls.default_outfit")

    @field_validator("expressions")
    @classmethod
    def validate_expressions(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("assistant.avatar_controls.expressions must not be empty")
        normalized: list[str] = []
        seen: set[str] = set()
        for expression in value:
            expression = _stripped_nonempty(
                expression,
                field_name="assistant.avatar_controls.expressions item",
            ).lower()
            if len(expression) > 64:
                raise ValueError("avatar expression names must be at most 64 characters")
            if expression in seen:
                raise ValueError(f"duplicate avatar expression: {expression!r}")
            seen.add(expression)
            normalized.append(expression)
        return normalized


class _AssistantConfig(_IdentityConfig):
    id: str = "default-agent"
    display_name: str = "Astra"
    system_prompt: str = "You are Astra, a local personal assistant."
    personality: _AssistantPersonalityConfig = Field(
        default_factory=_AssistantPersonalityConfig
    )
    avatar_controls: _AvatarControlsConfig = Field(default_factory=_AvatarControlsConfig)

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: str) -> str:
        value = _stripped_nonempty(value, field_name="assistant.system_prompt")
        if len(value) > 100_000:
            raise ValueError("assistant.system_prompt must be at most 100000 characters")
        return value


class _OrchestratorConfig(_StrictConfigModel):
    summary_trigger: int = Field(default=10, gt=0)
    generation_deadline_s: float = Field(default=600.0, gt=0)
    recovery_deadline_s: float = Field(default=180.0, gt=0)
    recovery_num_predict: int = Field(default=192, gt=0)


class _ContextConfig(_StrictConfigModel):
    history_limit: int = Field(default=6, gt=0)
    injected_memory_limit: int = Field(default=5, gt=0)
    integration_context_limit: int = Field(default=4000, gt=0)


class _BeliefsConfig(_StrictConfigModel):
    enabled: bool = True
    processing_mode: Literal["disabled", "observer", "react_tool"] = "disabled"
    timezone: str = "UTC"
    max_candidates: int = Field(default=4, gt=0)
    max_existing_beliefs: int = Field(default=24, gt=0)
    max_snapshot_chars: int = Field(default=2000, gt=0)
    max_disambiguating_context_chars: int = Field(default=1000, ge=0)
    max_generation_tokens: int = Field(default=384, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)
    max_expiry_days: int = Field(default=90, gt=0)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("must be a valid IANA timezone") from exc
        return value

    @model_validator(mode="after")
    def validate_enabled_mode(self):
        if not self.enabled and self.processing_mode != "disabled":
            raise ValueError(
                "beliefs.enabled=false requires beliefs.processing_mode=disabled"
            )
        return self


class _AutonomyConfig(_StrictConfigModel):
    enabled: bool = False
    max_chain_events: int = Field(default=20, gt=0)
    max_chain_age_s: float = Field(default=1800.0, gt=0)
    max_queue_size: int = Field(default=256, gt=0)
    max_tool_steps: int = Field(default=5, gt=0)
    global_llm_concurrency: int = Field(default=1, gt=0)
    approval_timeout_s: float = Field(default=300.0, gt=0)
    recent_context_limit: int = Field(default=4000, gt=0)


class _NativeAudioConfig(_StrictConfigModel):
    payload_field: Literal["images", "audios"] = "images"
    prompt_text: str = "Please answer the user's spoken audio."
    display_text: str = "Voice message"
    convert_to_wav: bool = True
    sample_rate: int = Field(default=16000, ge=8000, le=192000)


class _VoiceInputConfig(_StrictConfigModel):
    path: Literal["stt", "native_audio"] = "stt"
    native_audio: _NativeAudioConfig = Field(default_factory=_NativeAudioConfig)


class _VisionWatchdogConfig(_StrictConfigModel):
    model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    device: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: str = "auto"
    max_new_tokens: int = Field(default=8, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)


class _RuntimeConfigSections(_StrictConfigModel):
    llm: _LLMConfig
    assistant: _AssistantConfig
    local_human: _LocalHumanConfig
    orchestrator: _OrchestratorConfig
    context: _ContextConfig
    beliefs: _BeliefsConfig
    autonomy: _AutonomyConfig
    voice_input: _VoiceInputConfig
    vision_watchdog: _VisionWatchdogConfig


def _format_config_error(exc: ValidationError) -> ValueError:
    issues = []
    for error in exc.errors(include_url=False, include_input=False)[:12]:
        location = ".".join(str(part) for part in error["loc"])
        issues.append(f"{location}: {error['msg']} [{error['type']}]")
    return ValueError("Invalid configuration: " + "; ".join(issues))


class Config:
    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH):
        self.path = Path(path).expanduser().resolve()
        with self.path.open("r") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("Invalid configuration: document root must be a mapping")
        unknown_sections = sorted(set(raw) - _CONFIG_SECTIONS, key=repr)
        if unknown_sections:
            raise ValueError(
                "Invalid configuration: unsupported top-level sections: "
                + ", ".join(repr(name) for name in unknown_sections)
            )
        for section_name, section_value in raw.items():
            if not isinstance(section_value, dict):
                raise ValueError(
                    f"Invalid configuration: {section_name} must be a mapping"
                )
        self.raw = raw

        # Core sections
        self.llm = self.raw.get("llm", {})
        self.assistant = self.raw.get("assistant", {})
        self.local_human = self.raw.get("local_human", {})

        # Integrations and temporary legacy tool configuration
        self.tools = self.raw.get("tools", {})
        self.integrations = self._load_integrations_config(
            self.raw.get("integrations"),
            self.tools,
        )

        # Orchestrator
        self.orchestrator = self.raw.get("orchestrator", {})

        # Context
        self.context = self.raw.get("context", {})

        self.beliefs = self._load_beliefs_config(self.raw.get("beliefs"))

        self.autonomy = self.raw.get("autonomy", {})

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
        self.voice_input = self.raw.get("voice_input", {})

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

        runtime_sections = {
            "llm": self.llm,
            "assistant": self.assistant,
            "local_human": self.local_human,
            "orchestrator": self.orchestrator,
            "context": self.context,
            "beliefs": self.beliefs,
            "autonomy": self.autonomy,
            "voice_input": self.raw.get("voice_input", {}),
            "vision_watchdog": self.raw.get("vision_watchdog", {}),
        }
        try:
            validated = _RuntimeConfigSections.model_validate(runtime_sections)
        except ValidationError as exc:
            raise _format_config_error(exc) from exc

        normalized = validated.model_dump(mode="python", exclude_none=True)
        for section_name, section_value in normalized.items():
            setattr(self, section_name, section_value)
            self.raw[section_name] = section_value

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
    def _load_beliefs_config(raw_beliefs: dict | None) -> dict:
        config = {}
        if isinstance(raw_beliefs, dict):
            if "extraction_enabled" in raw_beliefs:
                raise ValueError(
                    "beliefs.extraction_enabled was replaced by beliefs.processing_mode; "
                    "valid values are: disabled, observer, react_tool"
                )
            config.update(raw_beliefs)
        valid_modes = {"disabled", "observer", "react_tool"}
        mode = config.get("processing_mode", "disabled")
        if mode not in valid_modes:
            raise ValueError(
                "beliefs.processing_mode must be one of: disabled, observer, react_tool"
            )
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
