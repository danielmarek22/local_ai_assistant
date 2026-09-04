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
            import app.server as server

            assert server.AUDIO_DIR == server.STATIC_DIR / "audio"
            assert not hasattr(server.app.state, "settings")
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
