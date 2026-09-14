import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from app.tts.gpt_sovits_tts import GPTSoVITSTTS


class GPTSoVITSTimeoutTests(unittest.TestCase):
    def test_connected_service_that_never_replies_times_out(self):
        release = threading.Event()
        entered = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                entered.set()
                release.wait(3)

            def log_message(self, *args):
                pass

        service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        try:
            engine = GPTSoVITSTTS(
                api_url=f"http://127.0.0.1:{service.server_port}/tts",
                connect_timeout=0.2, read_timeout=0.05,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "speech.wav"
                with self.assertRaisesRegex(RuntimeError, "Failed to synthesize"):
                    engine.synthesize("Hello", path)
                self.assertTrue(entered.is_set())
                self.assertFalse(path.exists())
        finally:
            release.set()
            service.shutdown()
            service.server_close()
            thread.join(1)

    def test_default_request_has_connect_and_read_deadlines(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "app.tts.gpt_sovits_tts.requests.get"
        ) as get:
            get.return_value.content = b"audio"
            GPTSoVITSTTS().synthesize("Hello", Path(directory) / "speech.wav")
            self.assertEqual(get.call_args.kwargs["timeout"], (5.0, 20.0))
