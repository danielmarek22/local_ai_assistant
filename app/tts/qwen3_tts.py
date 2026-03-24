from pathlib import Path
import logging

from app.tts.base import TTS


logger = logging.getLogger(__name__)


class Qwen3TTS(TTS):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cuda:0",
        speaker: str = "Ryan",
        language: str = "English",
        ref_audio_path: Path | None = None,
        ref_text: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.speaker = speaker
        self.language = language
        self.ref_audio_path = Path(ref_audio_path) if ref_audio_path else None
        self.ref_text = ref_text
        self.uses_custom_voice = "customvoice" in model_id.lower()

        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel

        self._soundfile = sf

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch.bfloat16,
        )

        self.voice_clone_prompt = None
        if not self.uses_custom_voice and self.ref_audio_path and self.ref_text:
            if self.ref_audio_path.exists():
                self._setup_voice_clone(self.ref_audio_path, self.ref_text)
            else:
                logger.warning(
                    "Qwen3 TTS reference audio not found at %s; voice cloning is disabled until the file is available.",
                    self.ref_audio_path,
                )

    def _setup_voice_clone(self, audio_path: Path, text: str) -> None:
        ref_wav, sr = self._soundfile.read(str(audio_path))
        self.voice_clone_prompt = self.model.create_voice_clone_prompt(
            ref_audio=(ref_wav, sr),
            ref_text=text,
        )

    def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.uses_custom_voice:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=self.speaker,
            )
        else:
            if not self.voice_clone_prompt:
                raise ValueError(
                    "Voice clone prompt is missing. Set both `tts.ref_audio_path` "
                    "and `tts.ref_text`, or switch to a Qwen CustomVoice model."
                )

            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=self.language,
                voice_clone_prompt=self.voice_clone_prompt,
            )

        self._soundfile.write(str(output_path), wavs[0], sr)
