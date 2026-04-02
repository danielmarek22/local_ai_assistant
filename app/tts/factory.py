from pathlib import Path

from app.tts.base import TTS


def _resolve_engine_name(tts_config: dict) -> str:
    return str(
        tts_config.get("engine")
        or tts_config.get("provider")
        or tts_config.get("backend")
        or "qwen3"
    ).lower()


def build_tts_engine(tts_config: dict) -> TTS:
    engine = _resolve_engine_name(tts_config)

    if engine == "qwen3":
        from app.tts.qwen3_tts import Qwen3TTS

        settings = tts_config.get("qwen3", tts_config)
        return Qwen3TTS(
            model_id=settings["model_id"],
            device=settings.get("device", "cuda:0"),
            speaker=settings.get("speaker", "Ryan"),
            language=settings.get("language", "English"),
            ref_audio_path=(
                Path(settings["ref_audio_path"])
                if settings.get("ref_audio_path")
                else None
            ),
            ref_text=settings.get("ref_text"),
        )

    if engine == "piper":
        from app.tts.piper_tts import PiperTTS

        settings = tts_config.get("piper", tts_config)
        return PiperTTS(
            model_path=Path(settings["model_path"]),
            use_cuda=settings.get("use_cuda", True),
        )

    if engine in ("gpt_sovits", "gpt-sovits", "sovits"):
        from app.tts.gpt_sovits_tts import GPTSoVITSTTS

        settings = tts_config.get("gpt_sovits", tts_config)
        return GPTSoVITSTTS(
            api_url=settings.get("api_url", "http://127.0.0.1:9880/tts"),
            ref_audio_path=str(settings.get("ref_audio_path", "")),
            prompt_text=settings.get("prompt_text", ""),
            text_lang=settings.get("text_lang", "en"),
            prompt_lang=settings.get("prompt_lang", "en"),
        )

    raise ValueError(
        f"Unsupported TTS engine '{engine}'. Supported engines: qwen3, piper, gpt_sovits."
    )
