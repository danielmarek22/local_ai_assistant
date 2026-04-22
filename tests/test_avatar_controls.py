import tempfile
import unittest
from pathlib import Path

from app.services.avatar_controls import (
    build_prompt_with_avatar_controls,
    discover_gesture_catalog,
    normalize_expressions,
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

    def test_normalize_expressions_defaults_when_empty(self):
        self.assertEqual(
            normalize_expressions([]),
            ["happy", "angry", "sad", "relaxed", "surprised", "neutral"],
        )

    def test_build_prompt_with_avatar_controls_appends_guidance(self):
        base_prompt = "You are Astra."
        prompt = build_prompt_with_avatar_controls(
            base_prompt,
            {"greeting": "/static/animations/Gestures/Greeting.fbx"},
            allowed_expressions=["happy", "neutral"],
        )

        self.assertIn("You are Astra.", prompt)
        self.assertIn("Avatar Expression Control", prompt)
        self.assertIn("Allowed emotions: happy, neutral", prompt)
        self.assertIn("Avatar Gesture Control", prompt)
        self.assertIn("[animation:name]", prompt)
        self.assertIn("greeting", prompt)

    def test_build_prompt_with_avatar_controls_keeps_expression_block_without_gestures(self):
        prompt = build_prompt_with_avatar_controls(
            "You are Astra.",
            {},
            allowed_expressions=["happy", "neutral"],
        )

        self.assertIn("Avatar Expression Control", prompt)
        self.assertIn("Allowed emotions: happy, neutral", prompt)
        self.assertNotIn("Avatar Gesture Control", prompt)


if __name__ == "__main__":
    unittest.main()
