import logging
from pathlib import Path
import requests

from app.tts.base import TTS

logger = logging.getLogger(__name__)

class GPTSoVITSTTS(TTS):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9880/tts",
        ref_audio_path: str = "",
        prompt_text: str = "",
        text_lang: str = "en",
        prompt_lang: str = "en",
    ):
        """
        Initializes the GPT-SoVITS TTS engine via its local API v2.
        """
        self.api_url = api_url
        self.ref_audio_path = ref_audio_path
        self.prompt_text = prompt_text
        self.text_lang = text_lang
        self.prompt_lang = prompt_lang

    def synthesize(self, text: str, output_path: Path) -> None:
        """
        Sends the text to the GPT-SoVITS local server and saves the resulting audio.
        """
        # Ensure the directory exists just like the other engines do
        output_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
        }

        try:
            logger.info("Sending text to GPT-SoVITS API...")
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)
                
        except requests.exceptions.RequestException as e:
            logger.error("GPT-SoVITS API request failed: %s", e)
            raise RuntimeError(f"Failed to synthesize speech using GPT-SoVITS: {e}")