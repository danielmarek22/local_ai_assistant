import re


SESSION_ID_MAX_LENGTH = 128
SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
_SESSION_ID_RE = re.compile(SESSION_ID_PATTERN)


def validate_session_id(value: object) -> str:
    if not isinstance(value, str) or _SESSION_ID_RE.fullmatch(value) is None:
        raise ValueError(
            "Invalid session ID: expected 1-128 ASCII letters, digits, underscores, "
            "or hyphens, beginning with a letter or digit"
        )
    return value
