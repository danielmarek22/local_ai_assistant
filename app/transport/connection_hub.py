from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

from app.core.assistant_state import AssistantState
from app.transport.websocket_protocol import ToolApprovalResponseFrame, encode_server_frame


MAX_PENDING_ASSISTANT_FRAMES = 128
_TERMINAL_ASSISTANT_FRAME_TYPES = frozenset(
    {"assistant_end", "assistant_retryable_error"}
)


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


@dataclass(frozen=True)
class SessionAssistantState:
    state: AssistantState = AssistantState.IDLE
    turn_id: str | None = None
    origin: str | None = None


class SessionConnectionHub:
    def __init__(self):
        self._connections: dict[str, _Connection] = {}
        self._by_websocket: dict[int, str] = {}
        self._approvals: dict[str, tuple[str, asyncio.Future[bool]]] = {}
        self._assistant_states: dict[str, SessionAssistantState] = {}
        self._pending_assistant_frames: dict[str, list[dict]] = {}

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

    def current_assistant_state(self, session_id: str) -> SessionAssistantState:
        return self._assistant_states.get(session_id, SessionAssistantState())

    async def finish_turn(self, connection_id: str, turn_id: str) -> None:
        connection = self._connections.get(connection_id)
        if connection is None:
            return

        current = self._assistant_states.get(connection.session_id)
        if current is not None and current.turn_id == turn_id:
            await self.broadcast(
                connection.session_id,
                {
                    "type": "assistant_state",
                    "state": AssistantState.IDLE,
                    "turn_id": turn_id,
                    "origin": connection.turn_origin,
                },
            )

        connection.turn_id = None
        connection.turn_origin = None

    async def send_websocket(self, websocket, payload: dict) -> None:
        connection_id = self._by_websocket.get(id(websocket))
        if connection_id is None:
            await websocket.send_text(encode_server_frame(payload))
            return
        connection = self._connections[connection_id]
        prepared_payload = self._with_connection_metadata(connection, payload)
        if str(prepared_payload.get("type", "")).startswith("assistant_"):
            await self.broadcast(connection.session_id, prepared_payload)
            return
        await self._send(connection, prepared_payload)

    async def broadcast(self, session_id: str, payload: dict) -> None:
        self._record_assistant_state(session_id, payload)
        targets = [
            item for item in self._connections.values() if item.session_id == session_id
        ]
        results = []
        if targets:
            results = await asyncio.gather(
                *(self._send(item, payload) for item in targets),
                return_exceptions=True,
            )
        if (
            str(payload.get("type", "")).startswith("assistant_")
            and payload.get("type") != "assistant_state"
            and not any(not isinstance(result, BaseException) for result in results)
        ):
            self._buffer_assistant_frame(session_id, payload)

    async def replay_pending(self, connection_id: str) -> None:
        connection = self._connections.get(connection_id)
        if connection is None:
            return

        pending = self._pending_assistant_frames.pop(connection.session_id, [])
        for index, payload in enumerate(pending):
            try:
                await self._send(connection, payload)
            except Exception:
                self._pending_assistant_frames[connection.session_id] = pending[index:]
                return

    def _buffer_assistant_frame(self, session_id: str, payload: dict) -> None:
        payload = dict(payload)
        if payload.get("type") in _TERMINAL_ASSISTANT_FRAME_TYPES:
            self._pending_assistant_frames[session_id] = [payload]
            return

        pending = self._pending_assistant_frames.setdefault(session_id, [])
        turn_id = payload.get("turn_id")
        if pending and turn_id and pending[-1].get("turn_id") != turn_id:
            pending.clear()
        pending.append(payload)
        del pending[:-MAX_PENDING_ASSISTANT_FRAMES]

    def _record_assistant_state(self, session_id: str, payload: dict) -> None:
        if payload.get("type") != "assistant_state":
            return

        state = AssistantState(payload.get("state"))
        turn_id = payload.get("turn_id")
        origin = payload.get("origin")
        turn_id = turn_id if isinstance(turn_id, str) and turn_id else None
        origin = origin if isinstance(origin, str) and origin else None

        if state == AssistantState.IDLE:
            current = self._assistant_states.get(session_id)
            if (
                current is not None
                and turn_id is not None
                and current.turn_id is not None
                and current.turn_id != turn_id
            ):
                return
            self._assistant_states.pop(session_id, None)
            return

        pending = self._pending_assistant_frames.get(session_id)
        if pending and turn_id and pending[-1].get("turn_id") != turn_id:
            self._pending_assistant_frames.pop(session_id, None)
        self._assistant_states[session_id] = SessionAssistantState(
            state=state,
            turn_id=turn_id,
            origin=origin,
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
        payload = self._with_connection_metadata(connection, payload)
        async with connection.send_lock:
            await connection.websocket.send_text(encode_server_frame(payload))

    @staticmethod
    def _with_connection_metadata(connection: _Connection, payload: dict) -> dict:
        payload = dict(payload)
        if str(payload.get("type", "")).startswith("assistant_"):
            if connection.turn_id:
                payload.setdefault("turn_id", connection.turn_id)
            if connection.turn_origin:
                payload.setdefault("origin", connection.turn_origin)
        return payload
