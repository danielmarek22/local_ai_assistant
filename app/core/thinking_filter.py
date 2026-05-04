import re


THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _trailing_marker_prefix_len(text: str, marker: str) -> int:
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


class ThinkingBlockFilter:
    """
    Strip streamed <think>...</think> blocks while tolerating chunk boundaries
    in the middle of the control tags.
    """

    def __init__(self):
        self._splitter = ThinkingBlockSplitter()

    def push(self, chunk: str) -> str:
        visible_text, _thinking_text = self._splitter.push(chunk)
        return visible_text

    def flush(self) -> str:
        visible_text, _thinking_text = self._splitter.flush()
        return visible_text


class ThinkingBlockSplitter:
    """
    Split streamed content into visible reply text and thinking text while
    tolerating chunk boundaries in the middle of the control tags.
    """

    def __init__(self):
        self._buffer = ""
        self._in_thinking_block = False

    def push(self, chunk: str) -> tuple[str, str]:
        if chunk:
            self._buffer += chunk
        return self._extract(force=False)

    def flush(self) -> tuple[str, str]:
        return self._extract(force=True)

    def _extract(self, force: bool) -> tuple[str, str]:
        visible_parts: list[str] = []
        thinking_parts: list[str] = []

        while self._buffer:
            if self._in_thinking_block:
                close_idx = self._buffer.find(THINK_CLOSE_TAG)
                if close_idx == -1:
                    if force:
                        thinking_parts.append(self._buffer)
                        self._buffer = ""
                        break

                    keep = _trailing_marker_prefix_len(self._buffer, THINK_CLOSE_TAG)
                    safe_end = len(self._buffer) - keep
                    if safe_end > 0:
                        thinking_parts.append(self._buffer[:safe_end])
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break

                if close_idx > 0:
                    thinking_parts.append(self._buffer[:close_idx])
                self._buffer = self._buffer[close_idx + len(THINK_CLOSE_TAG):]
                self._in_thinking_block = False
                continue

            open_idx = self._buffer.find(THINK_OPEN_TAG)
            if open_idx == -1:
                if force:
                    visible_parts.append(self._buffer)
                    self._buffer = ""
                    break

                keep = _trailing_marker_prefix_len(self._buffer, THINK_OPEN_TAG)
                safe_end = len(self._buffer) - keep
                if safe_end > 0:
                    visible_parts.append(self._buffer[:safe_end])
                    self._buffer = self._buffer[safe_end:]
                break

            if open_idx > 0:
                visible_parts.append(self._buffer[:open_idx])

            self._buffer = self._buffer[open_idx + len(THINK_OPEN_TAG):]
            self._in_thinking_block = True

        return "".join(visible_parts), "".join(thinking_parts)


def strip_complete_thinking_blocks(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub(" ", text)
    cleaned = cleaned.replace(THINK_OPEN_TAG, " ").replace(THINK_CLOSE_TAG, " ")
    return cleaned
