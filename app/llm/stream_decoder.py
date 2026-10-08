"""NDJSON decoding with the existing visible and buffered validation policies.

Visible streams expose chunks immediately and retain native decoding errors.
Buffered generations validate each chunk before accumulating an atomic result.
Completion markers and aggregate buffer limits remain the consumer's concern.
"""

import json

from .base import InferenceFailure


MAX_BUFFERED_LINE_BYTES = 512_000


def decode_chunk(line: bytes):
    """Decode a nonempty NDJSON line without changing its shape or errors."""
    return json.loads(line.decode("utf-8"))


def decode_buffered_chunk(line: bytes) -> dict:
    """Decode and validate a buffered chunk using typed inference failures."""
    if len(line) > MAX_BUFFERED_LINE_BYTES:
        raise InferenceFailure(
            "malformed_response",
            "Ollama stream chunk exceeded the bounded line size.",
        )
    try:
        chunk = decode_chunk(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceFailure(
            "malformed_response",
            "Ollama returned malformed streamed JSON.",
        ) from exc
    if not isinstance(chunk, dict) or not isinstance(chunk.get("message", {}), dict):
        raise InferenceFailure(
            "malformed_response",
            "Ollama returned an invalid streamed response shape.",
        )
    return chunk
