import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.perception.keys import PerceptionKey
from app.perception.state import PerceptionState
from app.services.perception_frames import PerceptionFrameController
from app.services.websocket_protocol import VisionFrame


PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AA"
    "AAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _frame(frame_type: str, name: str = "frame.png") -> VisionFrame:
    return VisionFrame(
        type=frame_type,
        attachment={
            "name": name,
            "mime_type": "image/png",
            "data": PNG_BASE64,
            "size_bytes": 69,
        },
    )


class FakeRuntime:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


class FakeWatchdog:
    def __init__(self, result: bool = True):
        self.result = result
        self.screen_calls = []
        self.webcam_calls = []

    async def evaluate_screen(self, image: str) -> bool:
        self.screen_calls.append(image)
        return self.result

    async def evaluate_webcam(self, image: str) -> bool:
        self.webcam_calls.append(image)
        return self.result


class PerceptionFrameControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_attachment_returns_image_without_updating_vision_state(self):
        orchestrator = SimpleNamespace(
            perception=PerceptionState(),
            autonomy_runtime=FakeRuntime(),
        )
        controller = PerceptionFrameController(
            orchestrator=orchestrator,
            watchdog=None,
            session_id="session-1",
            connection_id="connection-1",
        )

        attachment = await controller.handle(_frame("user_attached_frame"))

        self.assertEqual(attachment.name, "frame.png")
        self.assertEqual(orchestrator.perception.snapshot(), {})
        self.assertEqual(orchestrator.autonomy_runtime.events, [])

    async def test_screen_event_is_persisted_published_and_deduplicated(self):
        runtime = FakeRuntime()
        watchdog = FakeWatchdog()
        orchestrator = SimpleNamespace(
            perception=PerceptionState(),
            autonomy_runtime=runtime,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            controller = PerceptionFrameController(
                orchestrator=orchestrator,
                watchdog=watchdog,
                session_id="session-1",
                connection_id="connection-1",
                event_attachment_root=Path(temp_dir),
                detection_interval_seconds=0,
                clock=lambda: 10.0,
                event_id_factory=lambda: "event-1",
            )

            frame = _frame("screen_frame", "../screen capture.png")
            await controller.handle(frame)
            await controller.handle(frame)

            self.assertEqual(len(watchdog.screen_calls), 1)
            self.assertEqual(len(runtime.events), 1)
            event = runtime.events[0]
            self.assertEqual(str(event.event), "vision__attention_detected")
            self.assertEqual(event.session_id, "session-1")
            self.assertEqual(event.payload["source"], "screen")
            self.assertEqual(len(event.attachments), 1)
            stored_path = Path(event.attachments[0].storage_path)
            self.assertEqual(stored_path.parent, Path(temp_dir) / "event-1")
            self.assertEqual(stored_path.name, "screen_capture.png")
            self.assertTrue(stored_path.is_file())

        snapshot = orchestrator.perception.snapshot()
        self.assertEqual(
            snapshot[PerceptionKey.SCREEN_SCENE.value]["source"],
            "screen",
        )

    async def test_unavailable_watchdog_still_updates_perception(self):
        orchestrator = SimpleNamespace(
            perception=PerceptionState(),
            autonomy_runtime=None,
        )
        controller = PerceptionFrameController(
            orchestrator=orchestrator,
            watchdog=None,
            session_id="session-1",
            connection_id="connection-1",
            clock=lambda: 10.0,
        )

        result = await controller.handle(_frame("webcam_frame"))

        self.assertIsNone(result)
        snapshot = orchestrator.perception.snapshot()
        self.assertEqual(
            snapshot[PerceptionKey.WEBCAM_SCENE.value]["source"],
            "webcam",
        )


if __name__ == "__main__":
    unittest.main()
