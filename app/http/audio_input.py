"""Voice input preparation without application state or connection ownership."""

import logging
import subprocess

from app.config import Config
from app.perception.attachments import AudioAttachment

logger = logging.getLogger("server")

VOICE_INPUT_STT = "stt"
VOICE_INPUT_NATIVE_AUDIO = "native_audio"
_PYAV_INVALID_DATA_ERRNO = "1094995529"


def voice_input_path(settings: Config) -> str:
    path = str(settings.voice_input.get("path", VOICE_INPUT_STT)).strip().lower()
    if path in {"native", "audio", "gemma4"}:
        return VOICE_INPUT_NATIVE_AUDIO
    if path in {VOICE_INPUT_STT, VOICE_INPUT_NATIVE_AUDIO}:
        return path
    logger.warning("Unknown voice_input.path=%r; falling back to %s", path, VOICE_INPUT_STT)
    return VOICE_INPUT_STT


def native_audio_config(settings: Config) -> dict:
    native_audio = settings.voice_input.get("native_audio", {})
    return native_audio if isinstance(native_audio, dict) else {}


def is_invalid_stt_audio_error(exc: Exception) -> bool:
    """Return True for decoder errors caused by incomplete or non-audio blobs."""
    exc_type = type(exc)
    if exc_type.__name__ != "InvalidDataError":
        return False

    module = getattr(exc_type, "__module__", "")
    if module and not module.startswith("av"):
        return False

    message = str(exc)
    return (
        _PYAV_INVALID_DATA_ERRNO in message
        or "Invalid data found when processing input" in message
    )


def convert_audio_to_wav(
    audio_bytes: bytes,
    *,
    sample_rate: int = 16000,
    timeout_s: float = 15.0,
) -> bytes:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Audio conversion failed: {stderr or 'ffmpeg exited unsuccessfully'}")
    return result.stdout


def build_native_audio_attachment(
    audio_bytes: bytes,
    settings: Config,
) -> AudioAttachment:
    native_audio = native_audio_config(settings)
    convert_to_wav = bool(native_audio.get("convert_to_wav", True))

    if convert_to_wav:
        sample_rate = int(native_audio.get("sample_rate", 16000))
        payload = convert_audio_to_wav(audio_bytes, sample_rate=sample_rate)
        return AudioAttachment.from_bytes(
            payload,
            name="voice.wav",
            mime_type="audio/wav",
        )

    return AudioAttachment.from_bytes(
        audio_bytes,
        name=str(native_audio.get("raw_name", "voice.webm")),
        mime_type=str(native_audio.get("raw_mime_type", "audio/webm")),
    )
