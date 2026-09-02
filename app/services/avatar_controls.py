import logging
import re
from pathlib import Path


logger = logging.getLogger("avatar_controls")
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
DEFAULT_EXPRESSIONS = (
    "happy",
    "angry",
    "sad",
    "relaxed",
    "surprised",
    "neutral",
)


def normalize_gesture_name(name: str) -> str:
    normalized = _NORMALIZE_RE.sub("_", name.strip().lower()).strip("_")
    return normalized


def normalize_outfit_name(name: str) -> str:
    return _NORMALIZE_RE.sub("_", name.strip().lower()).strip("_")


def discover_outfit_catalog(avatars_dir: Path | None = None) -> dict[str, str]:
    static_dir = Path(__file__).resolve().parents[2] / "static"
    base_dir = avatars_dir or (static_dir / "avatars")
    catalog: dict[str, str] = {}

    if base_dir.exists():
        for path in sorted(base_dir.glob("*.vrm")):
            outfit_name = normalize_outfit_name(path.stem)
            if not outfit_name:
                logger.debug("Skipping outfit file with empty normalized name: %s", path.name)
                continue
            if outfit_name in catalog:
                raise ValueError(
                    f"Duplicate normalized outfit name {outfit_name!r}: "
                    f"{catalog[outfit_name]!r} and {path.name!r}"
                )
            catalog[outfit_name] = f"/static/avatars/{path.name}"

    legacy_avatar = static_dir / "avatar.vrm"
    if avatars_dir is None and "default" not in catalog and legacy_avatar.is_file():
        catalog["default"] = "/static/avatar.vrm"

    logger.info("Discovered %d avatar outfits", len(catalog))
    return catalog


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


def normalize_expressions(expressions: list[str] | tuple[str, ...] | None) -> list[str]:
    if not expressions:
        return list(DEFAULT_EXPRESSIONS)

    normalized: list[str] = []
    seen: set[str] = set()

    for item in expressions:
        value = str(item).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    return normalized or list(DEFAULT_EXPRESSIONS)


def build_prompt_with_avatar_controls(
    system_prompt: str,
    gesture_catalog: dict[str, str],
    allowed_expressions: list[str] | tuple[str, ...] | None = None,
) -> str:
    sections: list[str] = [system_prompt]
    expressions = normalize_expressions(allowed_expressions)
    expression_names = ", ".join(expressions)
    expression_block = (
        "## Avatar Expression Control\n"
        "You may embed expression tags within your response to control the avatar's face. "
        "These tags are filtered out before speech.\n"
        "- Formats: [state:emotion], [expression:emotion]\n"
        f"- Allowed emotions: {expression_names}\n"
        "- Usage: 0-2 tags per message\n"
        "- Guideline: Prefer keeping one expression unless the tone clearly changes\n"
        "- Rule: Place tags where emotional tone shifts. Do not explain or reference tags."
    )
    sections.append(expression_block)

    if gesture_catalog:
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
        sections.append(gesture_block)

    return "\n\n".join(section for section in sections if section)
