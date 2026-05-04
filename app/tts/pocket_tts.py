import logging
from pathlib import Path
import scipy.io.wavfile
from pocket_tts import TTSModel
from app.tts.base import TTS

logger = logging.getLogger(__name__)

class PocketTTSWrapper(TTS):
    def __init__(self, voice_state_path: str = "app/tts/escoffier.safetensors"):
        logger.info("🧠 Loading Pocket TTS Engine into system RAM...")
        # We load the model ONCE when the Uvicorn app starts up
        self.model = TTSModel.load_model()
        
        logger.info("👄 Loading Escoffier voice profile...")
        # We load the baked safetensors file ONCE
        self.voice_state = self.model.get_state_for_audio_prompt(voice_state_path)

    def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("⚡ Generating native CPU audio...")
            
            # Generate the audio tensor directly in memory!
            audio_tensor = self.model.generate_audio(self.voice_state, text)
            
            # Save the tensor to a .wav file for the frontend to grab
            scipy.io.wavfile.write(
                str(output_path), 
                self.model.sample_rate, 
                audio_tensor.numpy()
            )
            
        except Exception as e:
            logger.error("Pocket TTS synthesis failed: %s", e)
            raise RuntimeError(f"Failed to synthesize speech: {e}")