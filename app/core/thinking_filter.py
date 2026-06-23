import re
import json
from dataclasses import dataclass
from typing import Any


THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

TOOL_CALL_OPEN_TAG = "<tool_call>"
TOOL_CALL_CLOSE_TAG = "</tool_call>"
MEMORY_WRITE_OPEN_TAG = "<memory_write>"
MEMORY_WRITE_CLOSE_TAG = "</memory_write>"


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


@dataclass(frozen=True)
class ThinkingDirective:
    kind: str
    payload: dict[str, Any]
    raw: str


class ThinkingDirectiveFilter:
    """
    Strip internal late-routing directives from thinking text while returning
    completed directives for the orchestrator to execute.
    """

    _OPEN_TAGS = {
        "tool_call": TOOL_CALL_OPEN_TAG,
        "memory_write": MEMORY_WRITE_OPEN_TAG,
    }
    _CLOSE_TAGS = {
        "tool_call": TOOL_CALL_CLOSE_TAG,
        "memory_write": MEMORY_WRITE_CLOSE_TAG,
    }

    def __init__(self):
        self._buffer = ""
        self._active_kind: str | None = None

    def push(self, chunk: str) -> tuple[str, list[ThinkingDirective]]:
        if chunk:
            self._buffer += chunk
        return self._extract(force=False)

    def flush(self) -> tuple[str, list[ThinkingDirective]]:
        return self._extract(force=True)

    def _extract(self, force: bool) -> tuple[str, list[ThinkingDirective]]:
        visible_parts: list[str] = []
        directives: list[ThinkingDirective] = []

        while self._buffer:
            if self._active_kind:
                close_tag = self._CLOSE_TAGS[self._active_kind]
                close_idx = self._buffer.find(close_tag)
                if close_idx == -1:
                    if force:
                        self._buffer = ""
                        self._active_kind = None
                    break

                raw = self._buffer[:close_idx].strip()
                payload = _extract_json_object(raw)
                if payload is not None:
                    directives.append(
                        ThinkingDirective(
                            kind=self._active_kind,
                            payload=payload,
                            raw=raw,
                        )
                    )
                self._buffer = self._buffer[close_idx + len(close_tag):]
                self._active_kind = None
                continue

            match = self._find_next_open_tag(self._buffer)
            if match is None:
                if force:
                    visible_parts.append(self._buffer)
                    self._buffer = ""
                    break

                keep = self._trailing_open_prefix_len(self._buffer)
                safe_end = len(self._buffer) - keep
                if safe_end > 0:
                    visible_parts.append(self._buffer[:safe_end])
                    self._buffer = self._buffer[safe_end:]
                break

            kind, start_idx, tag = match
            if start_idx > 0:
                visible_parts.append(self._buffer[:start_idx])
            self._buffer = self._buffer[start_idx + len(tag):]
            self._active_kind = kind

        return "".join(visible_parts), directives

    def _find_next_open_tag(self, text: str) -> tuple[str, int, str] | None:
        matches = [
            (kind, idx, tag)
            for kind, tag in self._OPEN_TAGS.items()
            if (idx := text.find(tag)) != -1
        ]
        if not matches:
            return None
        return min(matches, key=lambda item: item[1])

    def _trailing_open_prefix_len(self, text: str) -> int:
        return max(
            (_trailing_marker_prefix_len(text, tag) for tag in self._OPEN_TAGS.values()),
            default=0,
        )

    def strip_directive_tags(self, text: str) -> str:
        """
        Remove any directive tags (<tool_call>...</tool_call> or <memory_write>...</memory_write>)
        from the given text, including bare JSON that matches the directive schema.
        Used to sanitize visible content that may have accidentally included directive
        syntax outside of thinking blocks.
        """
        # Remove complete directive blocks with tags
        for open_tag, close_tag in zip(self._OPEN_TAGS.values(), self._CLOSE_TAGS.values()):
            pattern = re.escape(open_tag) + r"[\s\S]*?" + re.escape(close_tag)
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Remove any lingering incomplete tags
        text = re.sub(re.escape(TOOL_CALL_OPEN_TAG), "", text, flags=re.IGNORECASE)
        text = re.sub(re.escape(TOOL_CALL_CLOSE_TAG), "", text, flags=re.IGNORECASE)
        text = re.sub(re.escape(MEMORY_WRITE_OPEN_TAG), "", text, flags=re.IGNORECASE)
        text = re.sub(re.escape(MEMORY_WRITE_CLOSE_TAG), "", text, flags=re.IGNORECASE)
        
        # Remove bare JSON objects matching directive schema (including nested objects)
        # This catches bare directive JSON without wrapper tags that leaked into visible content
        text = self._strip_bare_directive_json(text)
        
        return text
    
    def _strip_bare_directive_json(self, text: str) -> str:
        """
        Remove JSON objects that match the directive schema (tool_call or memory_write).
        Handles nested objects properly using a simple state machine.
        """
        result = []
        i = 0
        while i < len(text):
            # Look for JSON object start
            if text[i] == '{':
                # Try to extract the JSON object
                json_end = self._find_json_object_end(text, i)
                if json_end is not None:
                    json_str = text[i:json_end+1]
                    # Check if this JSON matches directive schema
                    if self._is_directive_json(json_str):
                        # Skip this JSON object (don't add to result)
                        i = json_end + 1
                        continue
                # Not a directive, keep the brace
                result.append(text[i])
                i += 1
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)
    
    def _find_json_object_end(self, text: str, start: int) -> int | None:
        """
        Find the closing brace of a JSON object starting at position start.
        Returns the index of the closing brace, or None if not found.
        """
        if start >= len(text) or text[start] != '{':
            return None
        
        depth = 0
        i = start
        in_string = False
        escape_next = False
        
        while i < len(text):
            char = text[i]
            
            if escape_next:
                escape_next = False
                i += 1
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                in_string = not in_string
                i += 1
                continue
            
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return i
            
            i += 1
        
        return None
    
    def _is_directive_json(self, json_str: str) -> bool:
        """
        Check if the given JSON string matches the directive schema.
        Returns True if it looks like a tool_call or memory_write directive.
        """
        try:
            obj = json.loads(json_str)
            if not isinstance(obj, dict):
                return False
            
            # Check for tool_call pattern: {"tool": "...", "kwargs": ...}
            if "tool" in obj and "kwargs" in obj:
                return True
            # Also support alternative formats: {"name": "...", "arguments": ...}
            if "name" in obj and "arguments" in obj:
                return True
            
            # Check for memory_write pattern: {"content": "...", "category": "...", ...}
            if "content" in obj and ("category" in obj or "importance" in obj):
                return True
            
            return False
        except (json.JSONDecodeError, TypeError):
            return False


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
