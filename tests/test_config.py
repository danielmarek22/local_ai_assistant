import tempfile
import unittest

import yaml
from pathlib import Path

from app.config import Config


class ConfigTests(unittest.TestCase):
    def test_local_human_has_stable_defaults(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump({}, config_file)
            config_file.flush()
            config = Config(config_file.name)
        self.assertEqual(config.local_human, {"id": "local-human", "display_name": "You"})

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

    def test_belief_storage_defaults_enabled_but_production_is_disabled(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump({}, config_file)
            config_file.flush()
            config = Config(config_file.name)

        self.assertTrue(config.beliefs["enabled"])
        self.assertEqual(config.beliefs["processing_mode"], "disabled")
        self.assertEqual(config.beliefs["max_candidates"], 4)
        self.assertEqual(config.beliefs["max_generation_tokens"], 384)
        self.assertEqual(config.beliefs["max_existing_beliefs"], 24)

    def test_all_belief_processing_modes_are_accepted(self):
        for mode in ("disabled", "observer", "react_tool"):
            with self.subTest(mode=mode), tempfile.NamedTemporaryFile(
                "w", suffix=".yaml"
            ) as config_file:
                yaml.safe_dump(
                    {"beliefs": {"enabled": True, "processing_mode": mode}},
                    config_file,
                )
                config_file.flush()
                config = Config(config_file.name)
            self.assertEqual(config.beliefs["processing_mode"], mode)

    def test_invalid_belief_processing_mode_fails(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump({"beliefs": {"processing_mode": "invalid"}}, config_file)
            config_file.flush()
            with self.assertRaisesRegex(ValueError, "disabled, observer, react_tool"):
                Config(config_file.name)

    def test_legacy_belief_extraction_key_fails_with_migration_guidance(self):
        for value in (True, False):
            with self.subTest(value=value), tempfile.NamedTemporaryFile(
                "w", suffix=".yaml"
            ) as config_file:
                yaml.safe_dump({"beliefs": {"extraction_enabled": value}}, config_file)
                config_file.flush()
                with self.assertRaisesRegex(ValueError, "was replaced.*processing_mode"):
                    Config(config_file.name)

    def test_disabled_belief_storage_rejects_active_producer(self):
        for mode in ("observer", "react_tool"):
            with self.subTest(mode=mode), tempfile.NamedTemporaryFile(
                "w", suffix=".yaml"
            ) as config_file:
                yaml.safe_dump(
                    {"beliefs": {"enabled": False, "processing_mode": mode}},
                    config_file,
                )
                config_file.flush()
                with self.assertRaisesRegex(ValueError, "enabled=false.*disabled"):
                    Config(config_file.name)

    def test_tracked_template_uses_processing_mode_without_legacy_key(self):
        template = yaml.safe_load(Path("app/config/assistant-template.yaml").read_text())
        self.assertEqual(template["beliefs"]["processing_mode"], "disabled")
        self.assertNotIn("extraction_enabled", template["beliefs"])

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
