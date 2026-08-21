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
        self.assertEqual(config.context["integration_context_limit"], 4000)

    def test_autonomy_defaults_are_bounded_and_disabled(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump({}, config_file)
            config_file.flush()
            config = Config(config_file.name)

        self.assertFalse(config.autonomy["enabled"])
        self.assertEqual(config.autonomy["max_chain_events"], 20)
        self.assertEqual(config.autonomy["global_llm_concurrency"], 1)

    def test_belief_storage_defaults_enabled_but_extraction_requires_opt_in(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump({}, config_file)
            config_file.flush()
            config = Config(config_file.name)

        self.assertTrue(config.beliefs["enabled"])
        self.assertFalse(config.beliefs["extraction_enabled"])
        self.assertEqual(config.beliefs["max_candidates"], 4)
        self.assertEqual(config.beliefs["max_generation_tokens"], 384)
        self.assertEqual(config.beliefs["max_existing_beliefs"], 24)

    def test_belief_extraction_can_be_enabled_explicitly(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump(
                {"beliefs": {"enabled": True, "extraction_enabled": True}},
                config_file,
            )
            config_file.flush()
            config = Config(config_file.name)

        self.assertTrue(config.beliefs["enabled"])
        self.assertTrue(config.beliefs["extraction_enabled"])

    def test_integration_config_defaults_memory_and_shell_enabled(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump({}, config_file)
            config_file.flush()
            config = Config(config_file.name)

        self.assertTrue(config.integrations["memory"]["enabled"])
        self.assertTrue(config.integrations["shell"]["enabled"])
        self.assertFalse(config.integrations["mindcraft"]["enabled"])
        self.assertEqual(config.integrations["mindcraft"]["url"], "http://localhost:8081")
        self.assertEqual(config.integrations["mindcraft"]["reconnect_max_delay_s"], 30.0)

    def test_legacy_web_config_is_used_only_without_new_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump({"tools": {"web": {"enabled": True, "base_url": "legacy"}}}, config_file)
            config_file.flush()
            with self.assertLogs("config", level="WARNING"):
                config = Config(config_file.name)

        self.assertEqual(config.integrations["web"]["base_url"], "legacy")

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump({
                "tools": {"web": {"enabled": True, "base_url": "legacy"}},
                "integrations": {"web": {"enabled": False, "base_url": "new"}},
            }, config_file)
            config_file.flush()
            config = Config(config_file.name)

        self.assertEqual(config.integrations["web"]["base_url"], "new")
        self.assertFalse(config.integrations["web"]["enabled"])

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
