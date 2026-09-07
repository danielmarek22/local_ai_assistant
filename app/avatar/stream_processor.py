import logging
import re

_DEFAULT_EXPRESSIONS = {
    "happy",
    "angry",
    "sad",
    "relaxed",
    "surprised",
    "neutral",
}
_EXPRESSION_ALIASES = {"expression", "state"}
_ANIMATION_ALIASES = {"animation", "gesture"}
_SPECIAL_CHARACTER_PATTERN = re.compile(r"[\\\[`~!]")

logger = logging.getLogger("stream_processor")


class StreamProcessor:
    def __init__(
        self,
        allowed_animations: set[str] | None = None,
        allowed_expressions: set[str] | None = None,
    ):
        self._buffer = ""
        self._markdown_mode: str | None = None
        self._markdown_delimiter = ""
        self.allowed_animations = {
            item.strip().lower() for item in (allowed_animations or set()) if item.strip()
        }
        expressions = (
            allowed_expressions
            if allowed_expressions is not None
            else _DEFAULT_EXPRESSIONS
        )
        self.allowed_expressions = {
            item.strip().lower() for item in expressions if item.strip()
        }

    def push(self, chunk: str) -> list[tuple[str, str]]:
        self._buffer += chunk
        return self._extract_events(force=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._extract_events(force=True)

    def _extract_events(self, force: bool) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        remainder = self._buffer

        while remainder:
            if self._markdown_mode is not None:
                closing_index = remainder.find(self._markdown_delimiter)
                if closing_index != -1:
                    closing_end = closing_index + len(self._markdown_delimiter)
                    events.append(("text", remainder[:closing_end]))
                    remainder = remainder[closing_end:]
                    self._markdown_mode = None
                    self._markdown_delimiter = ""
                    continue

                if force:
                    events.append(("text", remainder))
                    remainder = ""
                    self._markdown_mode = None
                    self._markdown_delimiter = ""
                    break

                held_length = self._delimiter_prefix_suffix_length(
                    remainder,
                    self._markdown_delimiter,
                )
                safe_end = len(remainder) - held_length
                if safe_end:
                    events.append(("text", remainder[:safe_end]))
                remainder = remainder[safe_end:]
                break

            special = _SPECIAL_CHARACTER_PATTERN.search(remainder)
            if special is None:
                events.append(("text", remainder))
                remainder = ""
                break

            if special.start() > 0:
                events.append(("text", remainder[:special.start()]))
                remainder = remainder[special.start():]

            marker = remainder[0]
            if marker == "\\":
                if len(remainder) == 1 and not force:
                    break
                escaped_end = min(2, len(remainder))
                events.append(("text", remainder[:escaped_end]))
                remainder = remainder[escaped_end:]
                continue

            if marker == "!":
                if len(remainder) == 1 and not force:
                    break
                if len(remainder) > 1 and remainder[1] == "[":
                    markdown_end = self._markdown_bracket_end(
                        remainder,
                        bracket_start=1,
                        force=force,
                    )
                    if markdown_end is None:
                        break
                    events.append(("text", remainder[:markdown_end]))
                    remainder = remainder[markdown_end:]
                else:
                    events.append(("text", marker))
                    remainder = remainder[1:]
                continue

            if marker in {"`", "~"}:
                run_length = len(remainder) - len(remainder.lstrip(marker))
                if run_length == len(remainder) and not force:
                    break
                delimiter = marker * run_length
                if marker == "`" or run_length >= 3:
                    self._markdown_mode = "fenced" if run_length >= 3 else "inline"
                    self._markdown_delimiter = delimiter
                events.append(("text", delimiter))
                remainder = remainder[run_length:]
                continue

            bracket_end = self._balanced_end(remainder, 0, "[", "]")
            if bracket_end is None:
                if force:
                    events.append(("text", remainder))
                    remainder = ""
                break

            if bracket_end == len(remainder) and not force:
                break

            markdown_end = self._markdown_bracket_end(
                remainder,
                bracket_start=0,
                force=force,
            )
            if markdown_end is None:
                break
            if markdown_end > bracket_end:
                events.append(("text", remainder[:markdown_end]))
                remainder = remainder[markdown_end:]
                continue

            tag_text = remainder[:bracket_end]
            event = self._parse_control_tag(tag_text)
            if event is not None:
                events.append(event)
            remainder = remainder[bracket_end:]

        self._buffer = remainder
        return events

    def _parse_control_tag(self, tag_text: str) -> tuple[str, str] | None:
        inner = tag_text[1:-1].strip()
        if ":" in inner:
            raw_marker, raw_value = inner.split(":", 1)
            marker = raw_marker.strip().lower()
            value = raw_value.strip().lower()

            if marker in _EXPRESSION_ALIASES:
                if value in self.allowed_expressions:
                    return ("expression", value)
                logger.debug("Ignoring invalid avatar expression tag %r", inner)
                return None

            if marker in _ANIMATION_ALIASES:
                if value in self.allowed_animations:
                    return ("animation", value)
                logger.debug("Ignoring invalid avatar animation tag %r", inner)
                return None

            logger.debug(
                "Ignoring bracketed value outside the avatar control schema: %r",
                inner,
            )
            return None

        bare_value = inner.lower()
        if bare_value in self.allowed_animations:
            return ("animation", bare_value)

        logger.debug(
            "Ignoring bracketed value outside the avatar control schema: %r",
            inner,
        )
        return None

    def _markdown_bracket_end(
        self,
        text: str,
        *,
        bracket_start: int,
        force: bool,
    ) -> int | None:
        bracket_end = self._balanced_end(text, bracket_start, "[", "]")
        if bracket_end is None:
            if force:
                return len(text)
            return None

        if bracket_end == len(text):
            return bracket_end

        suffix_marker = text[bracket_end]
        if suffix_marker not in {"(", "["}:
            return bracket_end
        if suffix_marker == "[" and ":" in text[bracket_start + 1:bracket_end - 1]:
            # Adjacent control-shaped tags are not Markdown reference links.
            # Each candidate must be validated independently so an invalid
            # prefix cannot shield a valid or malicious second candidate.
            return bracket_end

        closing_marker = ")" if suffix_marker == "(" else "]"
        suffix_end = self._balanced_end(
            text,
            bracket_end,
            suffix_marker,
            closing_marker,
        )
        if suffix_end is not None:
            return suffix_end
        if force:
            return len(text)
        return None

    @staticmethod
    def _balanced_end(text: str, start: int, opening: str, closing: str) -> int | None:
        depth = 0
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
                if depth == 0:
                    return index + 1
        return None

    @staticmethod
    def _delimiter_prefix_suffix_length(text: str, delimiter: str) -> int:
        max_length = min(len(text), len(delimiter) - 1)
        for length in range(max_length, 0, -1):
            if text.endswith(delimiter[:length]):
                return length
        return 0
