import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_MANIFEST = PROJECT_ROOT / "requirements.txt"
RUNTIME_GROUPS = (
    PROJECT_ROOT / "requirements-core.txt",
    PROJECT_ROOT / "requirements-media.txt",
    PROJECT_ROOT / "requirements-integrations.txt",
)
EXPECTED_OWNERS = {
    "requirements-core.txt": {
        "chromadb", "emoji", "fastapi", "jsonschema", "pillow", "pydantic",
        "pyyaml", "requests", "rich", "sentence-transformers", "torch",
        "transformers", "uvicorn",
    },
    "requirements-media.txt": {
        "faster-whisper", "piper-tts", "pocket-tts", "scipy",
    },
    "requirements-integrations.txt": {"python-socketio"},
}
EXACT_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_.,-]+\])?==[^\s]+$"
)


def _requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _requirement_name(requirement: str) -> str:
    return requirement.split("==", 1)[0].split("[", 1)[0].lower().replace("_", "-")


class DependencyManifestTests(unittest.TestCase):
    def test_full_install_includes_each_runtime_group_once(self):
        self.assertEqual(
            _requirements(FULL_MANIFEST),
            [f"-r {path.name}" for path in RUNTIME_GROUPS],
        )

    def test_runtime_groups_are_exact_pinned_and_disjoint(self):
        owners: dict[str, str] = {}
        for path in RUNTIME_GROUPS:
            for requirement in _requirements(path):
                match = EXACT_REQUIREMENT.fullmatch(requirement)
                self.assertIsNotNone(
                    match,
                    f"{path.name} must exact-pin {requirement!r}",
                )
                normalized_name = match.group("name").lower().replace("_", "-")
                self.assertNotIn(
                    normalized_name,
                    owners,
                    f"{normalized_name} is owned by both {owners.get(normalized_name)} and {path.name}",
                )
                owners[normalized_name] = path.name

        self.assertEqual(
            {
                path.name: {_requirement_name(item) for item in _requirements(path)}
                for path in RUNTIME_GROUPS
            },
            EXPECTED_OWNERS,
        )

    def test_removed_transitive_runtimes_are_not_direct_requirements(self):
        names = {
            _requirement_name(requirement)
            for path in RUNTIME_GROUPS
            for requirement in _requirements(path)
        }
        self.assertNotIn("onnxruntime-gpu", names)
        self.assertNotIn("soundfile", names)


if __name__ == "__main__":
    unittest.main()
