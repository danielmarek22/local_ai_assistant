import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"


def _application_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return {module for module in imports if module.startswith("app.")}


class PackageBoundaryTests(unittest.TestCase):
    def test_feature_packages_do_not_import_transport(self):
        violations = []
        for path in APP_ROOT.rglob("*.py"):
            if path.parent == APP_ROOT / "transport" or path == APP_ROOT / "server.py":
                continue
            if any(
                module == "app.transport" or module.startswith("app.transport.")
                for module in _application_imports(path)
            ):
                violations.append(str(path.relative_to(PROJECT_ROOT)))

        self.assertEqual(violations, [])

    def test_transport_does_not_import_feature_implementations(self):
        forbidden_prefixes = (
            "app.avatar",
            "app.autonomy",
            "app.beliefs",
            "app.integrations",
            "app.knowledge",
            "app.llm",
            "app.memory",
            "app.perception",
            "app.storage",
            "app.stt",
            "app.tools",
            "app.tts",
        )
        violations = []
        for path in (APP_ROOT / "transport").glob("*.py"):
            for module in _application_imports(path):
                if module.startswith(forbidden_prefixes):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)} imports {module}"
                    )

        self.assertEqual(violations, [])

    def test_services_namespace_is_retired(self):
        service_sources = sorted((APP_ROOT / "services").glob("*.py"))
        stale_imports = []
        for root in (APP_ROOT, PROJECT_ROOT / "tests"):
            for path in root.rglob("*.py"):
                for module in _application_imports(path):
                    if module == "app.services" or module.startswith("app.services."):
                        stale_imports.append(
                            f"{path.relative_to(PROJECT_ROOT)} imports {module}"
                        )

        self.assertEqual(service_sources, [])
        self.assertEqual(stale_imports, [])


if __name__ == "__main__":
    unittest.main()
