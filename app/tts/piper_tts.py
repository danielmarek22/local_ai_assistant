from pathlib import Path
import wave
from piper import PiperVoice
from piper.config import SynthesisConfig

from app.tts.base import TTS


class PiperTTS(TTS):
    def __init__(
        self,
        model_path: Path,
        use_cuda: bool = True,
        synthesis_config: SynthesisConfig | None = None,
    ):
        self.model_path = model_path
        self.voice = PiperVoice.load(
            str(model_path),
            use_cuda=use_cuda,
        )

        self.synthesis_config = synthesis_config or SynthesisConfig(
            volume=0.5,
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            normalize_audio=True,
        )

    def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_path), "wb") as wav_file:
            self.voice.synthesize_wav(
                text,
                wav_file,
                syn_config=self.synthesis_config,
            )
