import re

MEMORY_PATTERNS = [
    r"\bremember that\b",
    r"\bremember this\b",
    r"\bplease remember\b",
    r"\bnote that\b",
    r"\bsave this\b",
]


def is_memory_command(text: str) -> bool:
    text = text.lower()
    return any(re.search(p, text) for p in MEMORY_PATTERNS)


def extract_memory_content(text: str) -> str:
    """
    Strips the trigger phrase and returns clean memory content.
    """
    lowered = text.lower()

    for pattern in MEMORY_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            start = match.end()
            return text[start:].strip(" :.-")

    return text.strip()
