from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_ROOT / "static"
DATA_DIR = APP_ROOT / "data"
DEFAULT_CONFIG_PATH = APP_ROOT / "app" / "config" / "assistant.yaml"


def resolve_app_path(path: str | Path) -> Path:
    """Resolve application-owned relative paths independently of process CWD."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return APP_ROOT / candidate
