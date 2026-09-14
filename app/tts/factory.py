from app.tts.base import TTS
from app.paths import resolve_app_path


def _resolve_engine_name(tts_config: dict) -> str:
    engine = str(tts_config.get("engine") or "pocket_tts").lower()
    return {
        "pocket-tts": "pocket_tts",
        "pocket": "pocket_tts",
        "gpt-sovits": "gpt_sovits",
        "sovits": "gpt_sovits",
    }.get(engine, engine)


def build_tts_engine(tts_config: dict) -> TTS:
    engine = _resolve_engine_name(tts_config)

    if engine == "piper":
        from app.tts.piper_tts import PiperTTS

        settings = tts_config.get("piper", tts_config)
        return PiperTTS(
            model_path=resolve_app_path(settings["model_path"]),
            use_cuda=settings.get("use_cuda", True),
        )

    if engine == "pocket_tts":
        from app.tts.pocket_tts import PocketTTSWrapper

        return PocketTTSWrapper()

    if engine == "gpt_sovits":
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
        f"Unsupported TTS engine '{engine}'. Supported engines: pocket_tts, piper, "
        "gpt_sovits."
    )
