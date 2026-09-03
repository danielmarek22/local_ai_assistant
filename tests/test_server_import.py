import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class ServerImportTests(unittest.TestCase):
    def test_import_is_cwd_independent_and_does_not_write_there(self):
        script = textwrap.dedent(
            """
            import sys
            import types

            fake_qwen = types.ModuleType("qwen_tts")
            fake_qwen.Qwen3TTSModel = type("FakeQwen3TTSModel", (), {})
            fake_soundfile = types.ModuleType("soundfile")
            fake_torch = types.ModuleType("torch")
            fake_torch.bfloat16 = object()
            sys.modules.update({
                "qwen_tts": fake_qwen,
                "soundfile": fake_soundfile,
                "torch": fake_torch,
            })

            import app.server as server

            assert server.AUDIO_DIR == server.APP_ROOT / "static" / "audio"
            assert server.config.path == server.APP_ROOT / "app" / "config" / "assistant.yaml"
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(APP_ROOT)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(temp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
