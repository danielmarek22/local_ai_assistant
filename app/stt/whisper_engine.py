# app/stt/whisper_engine.py
from faster_whisper import WhisperModel
from dataclasses import dataclass
import io

@dataclass
class TranscriptionResult:
    text: str
    language: str
    avg_log_prob: float  # confidence proxy

class WhisperSTTEngine:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        vad_filter: bool = True,
        vad_parameters: dict | None = None,
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {"min_silence_duration_ms": 300}

    def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        import io
        segments, info = self.model.transcribe(
            io.BytesIO(audio_bytes),
            beam_size=5,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            avg_log_prob=info.language_probability,
        )
