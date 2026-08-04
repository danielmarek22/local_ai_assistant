import tempfile
import unittest

import yaml

from app.config import Config


class ConfigTests(unittest.TestCase):
    def test_context_uses_injected_memory_limit(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump(
                {
                    "context": {
                        "history_limit": 8,
                        "injected_memory_limit": 7,
                    }
                },
                config_file,
            )
            config_file.flush()

            config = Config(config_file.name)

        self.assertEqual(config.context["history_limit"], 8)
        self.assertEqual(config.context["injected_memory_limit"], 7)

    def test_voice_input_defaults_to_stt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump({}, config_file)
            config_file.flush()

            config = Config(config_file.name)

        self.assertEqual(config.voice_input["path"], "stt")
        self.assertEqual(config.voice_input["native_audio"]["payload_field"], "images")
        self.assertTrue(config.voice_input["native_audio"]["convert_to_wav"])

    def test_voice_input_merges_native_audio_overrides(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump(
                {
                    "voice_input": {
                        "path": "native_audio",
                        "native_audio": {
                            "payload_field": "audios",
                            "sample_rate": 24000,
                        },
                    }
                },
                config_file,
            )
            config_file.flush()

            config = Config(config_file.name)

        self.assertEqual(config.voice_input["path"], "native_audio")
        self.assertEqual(config.voice_input["native_audio"]["payload_field"], "audios")
        self.assertEqual(config.voice_input["native_audio"]["sample_rate"], 24000)
        self.assertEqual(
            config.voice_input["native_audio"]["prompt_text"],
            "Please answer the user's spoken audio.",
        )


if __name__ == "__main__":
    unittest.main()
