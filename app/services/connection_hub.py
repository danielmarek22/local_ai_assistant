from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

from app.services.websocket_protocol import ToolApprovalResponseFrame


def parse_approval_decision(
    payload: Mapping[str, object] | ToolApprovalResponseFrame,
) -> bool:
    """Return a decision only when the wire value is a JSON Boolean."""
    if isinstance(payload, ToolApprovalResponseFrame):
        return payload.approved
    approved = payload.get("approved")
    if type(approved) is not bool:
        raise ValueError("Approval response field 'approved' must be a Boolean")
    return approved


@dataclass
class _Connection:
    session_id: str
    connection_id: str
    websocket: object
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_active: float = field(default_factory=time.monotonic)
    turn_id: str | None = None
    turn_origin: str | None = None


class SessionConnectionHub:
    def __init__(self):
        self._connections: dict[str, _Connection] = {}
        self._by_websocket: dict[int, str] = {}
        self._approvals: dict[str, tuple[str, asyncio.Future[bool]]] = {}

    def register(self, session_id: str, connection_id: str, websocket) -> None:
        connection = _Connection(session_id, connection_id, websocket)
        self._connections[connection_id] = connection
        self._by_websocket[id(websocket)] = connection_id

    def unregister(self, connection_id: str) -> None:
        connection = self._connections.pop(connection_id, None)
        if connection is not None:
            self._by_websocket.pop(id(connection.websocket), None)
        for approval_id, (owner, future) in list(self._approvals.items()):
            if owner == connection_id:
                if not future.done():
                    future.set_result(False)
                self._approvals.pop(approval_id, None)

    def touch(self, connection_id: str) -> None:
        connection = self._connections.get(connection_id)
        if connection is not None:
            connection.last_active = time.monotonic()

    def has_session(self, session_id: str) -> bool:
        return any(item.session_id == session_id for item in self._connections.values())

    def set_turn(self, connection_id: str, turn_id: str | None, origin: str | None) -> None:
        connection = self._connections.get(connection_id)
        if connection is not None:
            connection.turn_id = turn_id
            connection.turn_origin = origin

    async def send_websocket(self, websocket, payload: dict) -> None:
        connection_id = self._by_websocket.get(id(websocket))
        if connection_id is None:
            await websocket.send_text(json.dumps(payload))
            return
        await self._send(self._connections[connection_id], payload)

    async def broadcast(self, session_id: str, payload: dict) -> None:
        targets = [
            item for item in self._connections.values() if item.session_id == session_id
        ]
        if targets:
            await asyncio.gather(
                *(self._send(item, payload) for item in targets),
                return_exceptions=True,
            )

    async def request_approval(
        self,
        session_id: str,
        request: dict,
        timeout_seconds: float,
    ) -> bool:
        candidates = [
            item for item in self._connections.values() if item.session_id == session_id
        ]
        if not candidates:
            return False
        connection = max(candidates, key=lambda item: item.last_active)
        approval_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._approvals[approval_id] = (connection.connection_id, future)
        payload = {
            "type": "tool_approval_request",
            "approval_id": approval_id,
            "tool": str(request.get("tool", "unknown")),
            "title": str(request.get("title", "Approve action?")),
            "reason": str(request.get("reason", "This action requires human approval.")),
            "detail_label": str(request.get("detail_label", "Details")),
            "detail": str(request.get("detail", "")),
            "timeout_seconds": timeout_seconds,
            "origin": "integration_event",
        }
        try:
            await self._send(connection, payload)
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except Exception:
            return False
        finally:
            self._approvals.pop(approval_id, None)

    def resolve_approval(
        self,
        connection_id: str,
        payload: Mapping[str, object] | ToolApprovalResponseFrame,
    ) -> bool:
        approval_id = (
            payload.approval_id
            if isinstance(payload, ToolApprovalResponseFrame)
            else payload.get("approval_id")
        )
        pending = self._approvals.get(str(approval_id))
        if pending is None or pending[0] != connection_id:
            return False
        future = pending[1]
        if not future.done():
            try:
                decision = parse_approval_decision(payload)
            except ValueError:
                decision = False
            future.set_result(decision)
        return True

    async def _send(self, connection: _Connection, payload: dict) -> None:
        payload = dict(payload)
        if str(payload.get("type", "")).startswith("assistant_"):
            if connection.turn_id:
                payload.setdefault("turn_id", connection.turn_id)
            if connection.turn_origin:
                payload.setdefault("origin", connection.turn_origin)
        async with connection.send_lock:
            await connection.websocket.send_text(json.dumps(payload))
