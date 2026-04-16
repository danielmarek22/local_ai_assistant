import logging
import re
from pathlib import Path


logger = logging.getLogger("gesture_catalog")
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_gesture_name(name: str) -> str:
    normalized = _NORMALIZE_RE.sub("_", name.strip().lower()).strip("_")
    return normalized


def discover_gesture_catalog(gestures_dir: Path | None = None) -> dict[str, str]:
    base_dir = gestures_dir or (Path(__file__).resolve().parents[2] / "static" / "animations" / "Gestures")
    catalog: dict[str, str] = {}

    if not base_dir.exists():
        logger.info("Gesture directory does not exist: %s", base_dir)
        return catalog

    for path in sorted(base_dir.glob("*.fbx")):
        gesture_name = normalize_gesture_name(path.stem)
        if not gesture_name:
            logger.debug("Skipping gesture file with empty normalized name: %s", path.name)
            continue

        if gesture_name in catalog:
            logger.warning(
                "Skipping duplicate gesture key '%s' from %s (already mapped to %s)",
                gesture_name,
                path.name,
                catalog[gesture_name],
            )
            continue

        catalog[gesture_name] = f"/static/animations/Gestures/{path.name}"

    logger.info("Discovered %d gesture animations", len(catalog))
    return catalog


def build_prompt_with_gesture_catalog(system_prompt: str, gesture_catalog: dict[str, str]) -> str:
    if not gesture_catalog:
        return system_prompt

    gesture_names = ", ".join(sorted(gesture_catalog.keys()))
    gesture_block = (
        "## Avatar Gesture Control\n"
        "You may embed gesture tags within your response to trigger one-shot body animations.\n"
        "- Format: [animation:name]\n"
        "- Alias: [gesture:name]\n"
        f"- Allowed names: {gesture_names}\n"
        "- Usage: 0-2 tags per message\n"
        "- Rule: Use gesture tags only when they add clear expressive value. Do not mention the tags."
    )
    return f"{system_prompt}\n\n{gesture_block}"
