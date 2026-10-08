import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.http.audio_input import (
    build_native_audio_attachment,
    is_invalid_stt_audio_error,
)


class InvalidSttAudioErrorTests(unittest.TestCase):
    def test_pyav_invalid_data_error_is_expected_stt_silence(self):
        invalid_data_error_type = type(
            "InvalidDataError",
            (Exception,),
            {"__module__": "av.error"},
        )
        exc = invalid_data_error_type(
            "[Errno 1094995529] Invalid data found when processing input: '<none>'"
        )

        self.assertTrue(is_invalid_stt_audio_error(exc))

    def test_other_errors_are_not_treated_as_stt_silence(self):
        self.assertFalse(
            is_invalid_stt_audio_error(RuntimeError("model unavailable"))
        )


class NativeAudioPreparationTests(unittest.TestCase):
    @patch("app.http.audio_input.subprocess.run")
    def test_attachment_payload_and_metadata_follow_conversion_settings(self, run):
        for config, expected_payload, expected_name, expected_mime in (
            ({}, b"wav", "voice.wav", "audio/wav"),
            ({"sample_rate": 24000}, b"wav", "voice.wav", "audio/wav"),
            ({"convert_to_wav": False}, b"raw", "voice.webm", "audio/webm"),
            ({"convert_to_wav": False, "raw_name": "clip.ogg", "raw_mime_type": "audio/ogg"},
             b"raw", "clip.ogg", "audio/ogg"),
        ):
            with self.subTest(config=config):
                run.reset_mock()
                run.return_value = SimpleNamespace(returncode=0, stdout=b"wav")
                settings = SimpleNamespace(voice_input={"native_audio": config})
                attachment = build_native_audio_attachment(b"raw", settings)
                self.assertEqual(attachment.as_bytes(), expected_payload)
                self.assertEqual(attachment.name, expected_name)
                self.assertEqual(attachment.mime_type, expected_mime)
                if config.get("convert_to_wav", True):
                    run.assert_called_once()
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index("-ar") + 1],
                                     str(config.get("sample_rate", 16000)))
                    self.assertEqual(command[command.index("-ac") + 1], "1")
                    self.assertEqual(run.call_args.kwargs["input"], b"raw")
                    self.assertEqual(run.call_args.kwargs["timeout"], 15.0)
                else:
                    run.assert_not_called()

    @patch("app.http.audio_input.subprocess.run")
    def test_conversion_failure_and_timeout_propagate(self, run):
        settings = SimpleNamespace(voice_input={})
        run.return_value = SimpleNamespace(returncode=1, stderr=b"invalid audio")
        with self.assertRaisesRegex(ValueError, "Audio conversion failed: invalid audio"):
            build_native_audio_attachment(b"raw", settings)
        run.side_effect = subprocess.TimeoutExpired("ffmpeg", 15.0)
        with self.assertRaises(subprocess.TimeoutExpired):
            build_native_audio_attachment(b"raw", settings)
