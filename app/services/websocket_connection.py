from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
import json
import logging
import time
from typing import Any
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from app.core.assistant_state import AssistantState
from app.services.connection_hub import parse_approval_decision
from app.services.websocket_protocol import (
    ToolApprovalResponseFrame,
    decode_client_frame,
    encode_server_frame,
)


logger = logging.getLogger("server")

MAX_WEBSOCKET_TEXT_BYTES = 15 * 1024 * 1024
MAX_WEBSOCKET_BINARY_BYTES = 10 * 1024 * 1024
TOOL_APPROVAL_TIMEOUT_SECONDS = 300.0


async def send_ws_payload(ws: WebSocket, payload: dict) -> None:
    application = getattr(ws, "app", None)
    hub = getattr(getattr(application, "state", None), "connection_hub", None)
    if hub is not None:
        await hub.send_websocket(ws, payload)
    else:
        await ws.send_text(encode_server_frame(payload))


async def flush_pending_chunks(ws: WebSocket, pending_chunks: list[str]) -> None:
    for chunk in pending_chunks:
        await send_ws_payload(ws, {
            "type": "assistant_chunk",
            "content": chunk,
        })
    pending_chunks.clear()


async def send_turn_error(ws: WebSocket, message: str) -> None:
    with suppress(Exception):
        await send_ws_payload(ws, {
            "type": "assistant_state",
            "state": AssistantState.IDLE,
        })

    with suppress(Exception):
        await send_ws_payload(ws, {
            "type": "assistant_end",
            "content": message,
        })


class WebSocketMessageTooLarge(ValueError):
    pass


class WebSocketMessageInbox:
    """Own WebSocket reads and replay messages deferred by nested protocol waits."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        max_text_bytes: int = MAX_WEBSOCKET_TEXT_BYTES,
        max_binary_bytes: int = MAX_WEBSOCKET_BINARY_BYTES,
    ):
        self.websocket = websocket
        self.max_text_bytes = max_text_bytes
        self.max_binary_bytes = max_binary_bytes
        self._deferred: deque[dict[str, Any]] = deque()

    async def receive(self) -> dict[str, Any]:
        if self._deferred:
            return self._deferred.popleft()
        return await self._receive_live()

    async def receive_live(self) -> dict[str, Any]:
        return await self._receive_live()

    def defer(self, message: dict[str, Any]) -> None:
        self._deferred.append(message)

    async def _receive_live(self) -> dict[str, Any]:
        message = await self.websocket.receive()
        try:
            self._validate_size(message)
        except WebSocketMessageTooLarge as exc:
            with suppress(Exception):
                await self.websocket.close(code=1009, reason=str(exc))
            raise
        return message

    def _validate_size(self, message: dict[str, Any]) -> None:
        text_payload = message.get("text")
        if text_payload is not None:
            text_bytes = len(text_payload.encode("utf-8"))
            if text_bytes > self.max_text_bytes:
                raise WebSocketMessageTooLarge(
                    f"Text frame exceeds the {self.max_text_bytes}-byte limit"
                )

        binary_payload = message.get("bytes")
        if binary_payload is not None and len(binary_payload) > self.max_binary_bytes:
            raise WebSocketMessageTooLarge(
                f"Binary frame exceeds the {self.max_binary_bytes}-byte limit"
            )


async def request_tool_approval(
    inbox: WebSocketMessageInbox,
    request: dict,
    connection_id: str,
    timeout_seconds: float = TOOL_APPROVAL_TIMEOUT_SECONDS,
) -> bool:
    ws = inbox.websocket
    approval_id = uuid.uuid4().hex
    tool_name = str(request.get("tool", "unknown"))
    title = str(request.get("title", "Approve action?"))
    reason = str(request.get("reason", "This action requires human approval."))
    detail_label = str(request.get("detail_label", "Details"))
    detail = str(request.get("detail", ""))

    logger.info("[%s] Requesting human approval for %s", connection_id, tool_name)
    await send_ws_payload(ws, {
        "type": "tool_approval_request",
        "approval_id": approval_id,
        "tool": tool_name,
        "title": title,
        "reason": reason,
        "detail_label": detail_label,
        "detail": detail,
        "timeout_seconds": timeout_seconds,
    })

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("[%s] Tool approval timed out for %s", connection_id, tool_name)
            return False

        raw_message = await asyncio.wait_for(inbox.receive_live(), timeout=remaining)
        if raw_message.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect()

        payload = None
        text_payload = raw_message.get("text")
        if text_payload is not None:
            try:
                candidate = decode_client_frame(text_payload)
            except ValueError:
                try:
                    malformed = json.loads(text_payload)
                except json.JSONDecodeError:
                    malformed = None
                if (
                    isinstance(malformed, dict)
                    and malformed.get("type") == "tool_approval_response"
                    and malformed.get("approval_id") == approval_id
                ):
                    logger.warning(
                        "[%s] Denying malformed approval response for %s",
                        connection_id,
                        tool_name,
                    )
                    return False
                candidate = None
            if (
                isinstance(candidate, ToolApprovalResponseFrame)
                and candidate.approval_id == approval_id
            ):
                payload = candidate

        if payload is None:
            inbox.defer(raw_message)
            logger.debug(
                "[%s] Deferred websocket message while awaiting tool approval",
                connection_id,
            )
            continue

        try:
            approved = parse_approval_decision(payload)
        except ValueError:
            logger.warning(
                "[%s] Denying malformed approval response for %s",
                connection_id,
                tool_name,
            )
            return False
        logger.info(
            "[%s] Human %s capability %s",
            connection_id,
            "approved" if approved else "denied",
            tool_name,
        )
        return approved
