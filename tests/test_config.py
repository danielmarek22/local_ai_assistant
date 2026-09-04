import tempfile
import unittest

import yaml
from pathlib import Path

from app.config import Config
from app.tts.factory import _resolve_engine_name, build_tts_engine


class ConfigTests(unittest.TestCase):
    def _load(self, payload):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            yaml.safe_dump(payload, config_file)
            config_file.flush()
            return Config(config_file.name)

    def test_document_root_and_sections_must_be_mappings(self):
        for payload, expected in (
            (["not", "a", "mapping"], "document root must be a mapping"),
            ({"context": []}, "context must be a mapping"),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, expected):
                self._load(payload)

    def test_unknown_top_level_and_runtime_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported top-level.*contex"):
            self._load({"contex": {"history_limit": 5}})
        with self.assertRaisesRegex(ValueError, "unsupported top-level.*planner"):
            self._load({"planner": {"mode": "rule"}})
        with self.assertRaisesRegex(ValueError, "context.typo.*extra_forbidden"):
            self._load({"context": {"typo": 5}})
        with self.assertRaisesRegex(ValueError, "voice_input.typo.*extra_forbidden"):
            self._load({"voice_input": {"typo": True}})

    def test_runtime_scalar_types_and_bounds_are_strict(self):
        invalid = (
            ({"autonomy": {"enabled": "false"}}, "autonomy.enabled"),
            ({"context": {"history_limit": 0}}, "context.history_limit"),
            ({"orchestrator": {"recovery_num_predict": True}}, "recovery_num_predict"),
            ({"voice_input": {"path": "native"}}, "voice_input.path"),
            ({"vision_watchdog": {"max_new_tokens": -1}}, "max_new_tokens"),
        )
        for payload, expected in invalid:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, expected):
                self._load(payload)

    def test_llm_and_identity_unknown_fields_are_rejected(self):
        invalid = (
            ({"llm": {"typo": True}}, "llm.typo"),
            ({"llm": {"generation": {"frequency_penalty": 1.0}}}, "frequency_penalty"),
            ({"llm": {"thinking": {"mode": "high"}}}, "thinking.mode"),
            ({"assistant": {"nickname": "Star"}}, "assistant.nickname"),
            ({"assistant": {"personality": {"tone": "warm"}}}, "personality.tone"),
            ({"local_human": {"name": "User"}}, "local_human.name"),
        )
        for payload, expected in invalid:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, expected):
                self._load(payload)

    def test_llm_types_ranges_and_aliases_are_strict(self):
        invalid = (
            ({"llm": {"backend": "openai"}}, "llm.backend"),
            ({"llm": {"timeout_s": "30"}}, "llm.timeout_s"),
            ({"llm": {"max_retries": -1}}, "llm.max_retries"),
            ({"llm": {"thinking": {"enabled": "yes"}}}, "thinking.enabled"),
            ({"llm": {"thinking": {"level": "maximum"}}}, "thinking.level"),
            ({"llm": {"generation": {"temperature": 2.1}}}, "temperature"),
            ({"llm": {"generation": {"top_p": 1.1}}}, "top_p"),
            ({"llm": {"generation": {"max_tokens": 0}}}, "max_tokens"),
            (
                {"llm": {"generation": {"max_tokens": 128, "num_predict": 128}}},
                "configure only one",
            ),
            (
                {"llm": {"generation": {"rep_pen": 1.1, "repeat_penalty": 1.1}}},
                "configure only one",
            ),
        )
        for payload, expected in invalid:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, expected):
                self._load(payload)

    def test_llm_host_is_a_normalized_http_origin(self):
        config = self._load({"llm": {"host": " https://ollama.example.test/ "}})
        self.assertEqual(config.llm["host"], "https://ollama.example.test")

        for host in (
            "ollama.example.test",
            "ftp://ollama.example.test",
            "http://user:secret@ollama.example.test",
            "http://ollama.example.test/api",
            "http://ollama.example.test?model=test",
        ):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "HTTP.*origin"):
                self._load({"llm": {"host": host}})

    def test_identity_values_are_normalized_and_bounded(self):
        config = self._load({
            "local_human": {"id": " user-1 ", "display_name": " User One "},
            "assistant": {
                "id": " agent:astra ",
                "display_name": " Astra ",
                "system_prompt": " Be helpful. ",
                "avatar_controls": {
                    "default_outfit": " default ",
                    "expressions": [" Happy ", "neutral"],
                },
            },
        })

        self.assertEqual(config.local_human, {"id": "user-1", "display_name": "User One"})
        self.assertEqual(config.assistant["id"], "agent:astra")
        self.assertEqual(config.assistant["display_name"], "Astra")
        self.assertEqual(config.assistant["system_prompt"], "Be helpful.")
        self.assertEqual(
            config.assistant["avatar_controls"]["expressions"],
            ["happy", "neutral"],
        )

    def test_invalid_identity_and_avatar_values_are_rejected(self):
        invalid = (
            ({"local_human": {"id": "../user"}}, "local_human.id"),
            ({"local_human": {"display_name": 123}}, "local_human.display_name"),
            ({"assistant": {"system_prompt": "  "}}, "system_prompt"),
            (
                {"assistant": {"avatar_controls": {"expressions": []}}},
                "expressions must not be empty",
            ),
            (
                {"assistant": {"avatar_controls": {"expressions": ["happy", "HAPPY"]}}},
                "duplicate avatar expression",
            ),
        )
        for payload, expected in invalid:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, expected):
                self._load(payload)

    def test_tts_engine_aliases_are_normalized(self):
        for alias, canonical in (
            ("pocket-tts", "pocket_tts"),
            ("pocket", "pocket_tts"),
            ("gpt-sovits", "gpt_sovits"),
            ("sovits", "gpt_sovits"),
        ):
            payload = {"tts": {"engine": alias}}
            if canonical == "gpt_sovits":
                payload["tts"]["gpt_sovits"] = {"ref_audio_path": "voice.wav"}
            with self.subTest(alias=alias):
                config = self._load(payload)
                self.assertEqual(config.tts["engine"], canonical)
                self.assertEqual(_resolve_engine_name({"engine": alias}), canonical)

    def test_removed_tts_layouts_fail_with_migration_guidance(self):
        invalid = (
            ({"tts": {"engine": "qwen3"}}, "no Qwen TTS engine is implemented"),
            ({"tts": {"qwen3": {"model_id": "old"}}}, "no Qwen TTS engine"),
            ({"tts": {"provider": "piper"}}, "provider.*replaced by tts.engine"),
            ({"tts": {"backend": "piper"}}, "backend.*replaced by tts.engine"),
            ({"tts": {"model_path": "voice.onnx"}}, "flat TTS settings.*tts.piper"),
        )
        for payload, expected in invalid:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, expected):
                self._load(payload)

        with self.assertRaisesRegex(ValueError, "Supported engines: pocket_tts"):
            build_tts_engine({"engine": "qwen3"})

    def test_tts_selected_engine_contracts_are_strict(self):
        invalid = (
            ({"tts": {"engine": "gpt_sovits"}}, "requires.*ref_audio_path"),
            (
                {
                    "tts": {
                        "engine": "gpt_sovits",
                        "gpt_sovits": {
                            "ref_audio_path": "voice.wav",
                            "api_url": "file:///tmp/tts",
                        },
                    }
                },
                "HTTP.*URL",
            ),
            ({"tts": {"piper": {"use_cuda": "false"}}}, "piper.use_cuda"),
            ({"tts": {"piper": {"model": "voice.onnx"}}}, "piper.model"),
        )
        for payload, expected in invalid:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, expected):
                self._load(payload)

    def test_stt_contract_normalizes_names_and_vad_defaults(self):
        config = self._load({
            "stt": {
                "model_size": " small ",
                "device": " cpu ",
                "compute_type": " int8 ",
                "vad_parameters": {"threshold": 0.6},
            }
        })

        self.assertEqual(config.stt["model_size"], "small")
        self.assertEqual(config.stt["device"], "cpu")
        self.assertEqual(config.stt["compute_type"], "int8")
        self.assertEqual(config.stt["vad_parameters"]["threshold"], 0.6)
        self.assertEqual(config.stt["vad_parameters"]["min_silence_duration_ms"], 300)

    def test_stt_types_ranges_and_nested_fields_are_strict(self):
        invalid = (
            ({"stt": {"enabled": "true"}}, "stt.enabled"),
            ({"stt": {"vad_filter": 1}}, "stt.vad_filter"),
            ({"stt": {"model_size": " "}}, "stt.model_size"),
            ({"stt": {"vad_parameters": {"threshold": 1.1}}}, "threshold"),
            ({"stt": {"vad_parameters": {"min_silence_duration_ms": -1}}}, "min_silence"),
            ({"stt": {"vad_parameters": {"unknown": 1}}}, "vad_parameters.unknown"),
            (
                {
                    "stt": {
                        "vad_parameters": {"threshold": 0.4, "neg_threshold": 0.5}
                    }
                },
                "neg_threshold must not exceed threshold",
            ),
        )
        for payload, expected in invalid:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, expected):
                self._load(payload)

    def test_belief_timezone_must_be_valid(self):
        with self.assertRaisesRegex(ValueError, "valid IANA timezone"):
            self._load({"beliefs": {"timezone": "Mars/Olympus"}})

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
        template_path = Path("app/config/assistant-template.yaml")
        template = yaml.safe_load(template_path.read_text())
        self.assertEqual(template["beliefs"]["processing_mode"], "disabled")
        self.assertNotIn("extraction_enabled", template["beliefs"])
        self.assertNotIn("planner", template)

        config = Config(template_path)
        self.assertEqual(config.llm["backend"], "ollama")
        self.assertEqual(config.assistant["display_name"], "Astra")

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
