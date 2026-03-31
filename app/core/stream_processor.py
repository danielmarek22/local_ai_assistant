import re


_AVATAR_EXPRESSION_PATTERN = re.compile(
    r"\[\s*(?:[^:\]]+\s*:\s*)?(happy|angry|sad|relaxed|surprised|neutral)\s*\]",
    re.IGNORECASE,
)


class StreamProcessor:
    def __init__(self):
        self._buffer = ""

    def push(self, chunk: str) -> list[tuple[str, str]]:
        self._buffer += chunk
        return self._extract_events(force=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._extract_events(force=True)

    def _extract_events(self, force: bool) -> list[tuple[str, str]]:
        events = []
        remainder = self._buffer

        while remainder:
            match = _AVATAR_EXPRESSION_PATTERN.search(remainder)
            if match:
                if match.start() > 0:
                    events.append(("text", remainder[:match.start()]))

                events.append(("expression", match.group(1).lower()))
                remainder = remainder[match.end():]
                continue

            if force:
                events.append(("text", remainder))
                remainder = ""
                break

            marker_start = self._find_incomplete_expression_start(remainder)
            if marker_start is None:
                events.append(("text", remainder))
                remainder = ""
                break

            if marker_start > 0:
                events.append(("text", remainder[:marker_start]))

            remainder = remainder[marker_start:]
            break

        self._buffer = remainder
        return events

    def _find_incomplete_expression_start(self, text: str) -> int | None:
        last_bracket = text.rfind("[")
        if last_bracket == -1:
            return None

        if "]" not in text[last_bracket:]:
            return last_bracket

        return None
