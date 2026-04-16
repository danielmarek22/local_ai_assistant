import re
import logging


_AVATAR_EXPRESSION_PATTERN = re.compile(
    r"\[\s*(?:[^:\]]+\s*:\s*)?(happy|angry|sad|relaxed|surprised|neutral)\s*\]",
    re.IGNORECASE,
)
_BRACKETED_TAG_PATTERN = re.compile(r"\[[^\[\]]+\]")
_ANIMATION_ALIASES = {"animation", "gesture"}

logger = logging.getLogger("stream_processor")


class StreamProcessor:
    def __init__(self, allowed_animations: set[str] | None = None):
        self._buffer = ""
        self.allowed_animations = {item.lower() for item in (allowed_animations or set())}

    def push(self, chunk: str) -> list[tuple[str, str]]:
        self._buffer += chunk
        return self._extract_events(force=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._extract_events(force=True)

    def _extract_events(self, force: bool) -> list[tuple[str, str]]:
        events = []
        remainder = self._buffer

        while remainder:
            match = _BRACKETED_TAG_PATTERN.search(remainder)
            if match:
                if match.start() > 0:
                    events.append(("text", remainder[:match.start()]))

                event = self._parse_control_tag(match.group(0))
                if event is not None:
                    events.append(event)
                remainder = remainder[match.end():]
                continue

            if force:
                events.append(("text", remainder))
                remainder = ""
                break

            marker_start = self._find_incomplete_tag_start(remainder)
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

    def _parse_control_tag(self, tag_text: str) -> tuple[str, str] | None:
        inner = tag_text[1:-1].strip()
        if ":" in inner:
            marker, raw_value = inner.split(":", 1)
            marker = marker.strip().lower()
            value = raw_value.strip().lower()

            if marker in _ANIMATION_ALIASES:
                if value in self.allowed_animations:
                    return ("animation", value)

                if value:
                    logger.debug("Ignoring unknown avatar animation tag %r", value)
                else:
                    logger.debug("Ignoring avatar animation tag with empty animation name")
                return None

        expression_match = _AVATAR_EXPRESSION_PATTERN.fullmatch(tag_text)
        if expression_match:
            return ("expression", expression_match.group(1).lower())

        # Fallback: allow bare gesture tags like [greeting] when they match
        # a known gesture key from the runtime catalog.
        bare_value = inner.lower()
        if bare_value in self.allowed_animations:
            return ("animation", bare_value)

        return ("text", tag_text)

    def _find_incomplete_tag_start(self, text: str) -> int | None:
        last_bracket = text.rfind("[")
        if last_bracket == -1:
            return None

        if "]" not in text[last_bracket:]:
            return last_bracket

        return None
