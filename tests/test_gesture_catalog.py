import tempfile
import unittest
from pathlib import Path

from app.services.gesture_catalog import (
    build_prompt_with_gesture_catalog,
    discover_gesture_catalog,
    normalize_gesture_name,
)


class GestureCatalogTests(unittest.TestCase):
    def test_normalize_gesture_name(self):
        self.assertEqual(normalize_gesture_name(" Greeting Pose "), "greeting_pose")
        self.assertEqual(normalize_gesture_name("Happy!"), "happy")

    def test_discover_gesture_catalog_from_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gestures_dir = Path(temp_dir)
            (gestures_dir / "Greeting.fbx").write_text("fake", encoding="utf-8")
            (gestures_dir / "Happy Pose.fbx").write_text("fake", encoding="utf-8")
            (gestures_dir / "README.txt").write_text("ignored", encoding="utf-8")

            catalog = discover_gesture_catalog(gestures_dir=gestures_dir)

        self.assertEqual(
            catalog,
            {
                "greeting": "/static/animations/Gestures/Greeting.fbx",
                "happy_pose": "/static/animations/Gestures/Happy Pose.fbx",
            },
        )

    def test_build_prompt_with_gesture_catalog_appends_guidance(self):
        base_prompt = "You are Astra."
        prompt = build_prompt_with_gesture_catalog(
            base_prompt,
            {"greeting": "/static/animations/Gestures/Greeting.fbx"},
        )

        self.assertIn("You are Astra.", prompt)
        self.assertIn("Avatar Gesture Control", prompt)
        self.assertIn("[animation:name]", prompt)
        self.assertIn("greeting", prompt)


if __name__ == "__main__":
    unittest.main()
