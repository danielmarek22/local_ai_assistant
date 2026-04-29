# app/stt/factory.py
from .whisper_engine import WhisperSTTEngine

def build_stt_engine(stt_config: dict) -> WhisperSTTEngine | None:
    if not stt_config.get("enabled", False):
        return None
    return WhisperSTTEngine(
        model_size=stt_config.get("model_size", "small"),
        device=stt_config.get("device", "cpu"),
        compute_type=stt_config.get("compute_type", "int8"),
    )