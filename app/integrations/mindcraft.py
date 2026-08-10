from __future__ import annotations

import json
import logging
import threading
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from app.integrations.contracts import (
    CapabilityId,
    ContextContribution,
    InvocationContext,
    RegisteredTool,
    ToolResult,
    ToolSpec,
    EventId,
    EventPublisher,
    EventSpec,
    IntegrationEvent,
    NotificationPolicy,
    ReplayPolicy,
)


logger = logging.getLogger("mindcraft_integration")


class MindcraftUnavailable(RuntimeError):
    pass


class MindcraftClient:
    """Attach-only Socket.IO client for a running Mindcraft mindserver."""

    def __init__(
        self,
        url: str,
        agent_name: str | None = None,
        connect_timeout: float = 3.0,
        reconnect_delay_s: float = 2.0,
        reconnect_max_delay_s: float = 30.0,
        recent_output_limit: int = 3,
        socket_client: Any | None = None,
    ):
        self.url = url.rstrip("/")
        self.agent_name = (agent_name or "").strip() or None
        self.connect_timeout = max(0.1, float(connect_timeout))
        self.reconnect_delay_s = max(0.1, float(reconnect_delay_s))
        self.reconnect_max_delay_s = max(
            self.reconnect_delay_s,
            float(reconnect_max_delay_s),
        )
        self._socket = socket_client or self._create_socket_client()
        self._lock = threading.RLock()
        self._emit_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._connection_failure_count = 0
        self._outage_warning_logged = False
        self._agents: list[dict[str, object]] = []
        self._states: dict[str, object] = {}
        self._recent_output: deque[tuple[str, object]] = deque(
            maxlen=max(1, int(recent_output_limit))
        )
        self._command_result_handler = None
        self._register_handlers()

    @staticmethod
    def _create_socket_client():
        try:
            import socketio
        except ImportError as exc:
            raise RuntimeError(
                "Mindcraft integration requires python-socketio[client]"
            ) from exc
        return socketio.Client(reconnection=True)

    def _register_handlers(self) -> None:
        self._socket.on("connect", self._on_connect)
        self._socket.on("disconnect", self._on_disconnect)
        self._socket.on("agents-status", self._on_agents_status)
        self._socket.on("state-update", self._on_state_update)
        self._socket.on("bot-output", self._on_bot_output)
        self._socket.on("command-result", self._on_command_result)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._connection_loop,
            name="mindcraft-client",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        try:
            if getattr(self._socket, "connected", False):
                self._socket.disconnect()
        except Exception:
            logger.exception("Failed to disconnect from Mindcraft")
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.connect_timeout + 1.0)

    def is_available(self) -> bool:
        with self._lock:
            if not self._connected:
                return False
            target = self._resolve_agent_locked()
            status = self._status_for_locked(target)
            return bool(
                status
                and status.get("in_game")
                and status.get("socket_connected")
            )

    def send_message(self, message: str) -> str:
        return self._dispatch(message)

    def send_command(self, command: str, invocation_id: str | None = None) -> str:
        if not command.startswith("!"):
            raise ValueError("Mindcraft commands must start with '!'")
        if invocation_id is None:
            return self._dispatch(command)
        return self._dispatch_command(command, invocation_id)

    def set_command_result_handler(self, handler) -> None:
        self._command_result_handler = handler

    def _dispatch_command(self, command: str, invocation_id: str) -> str:
        target = self._ready_target()
        payload = {
            "request_id": invocation_id,
            "from": "local_assistant",
            "command": command,
        }
        try:
            with self._emit_lock:
                self._socket.emit("execute-command", (target, payload))
        except Exception as exc:
            raise MindcraftUnavailable(
                f"Could not dispatch a command to Mindcraft agent '{target}'."
            ) from exc
        return target

    def _dispatch(self, message: str) -> str:
        target = self._ready_target()

        payload = {"from": "local_assistant", "message": message}
        try:
            with self._emit_lock:
                self._socket.emit("send-message", (target, payload))
        except Exception as exc:
            raise MindcraftUnavailable(
                f"Could not dispatch an instruction to Mindcraft agent '{target}'."
            ) from exc
        return target

    def _ready_target(self) -> str:
        with self._lock:
            if not self._connected:
                raise MindcraftUnavailable("Mindcraft server is disconnected.")
            target = self._resolve_agent_locked()
            status = self._status_for_locked(target)
            if target is None:
                raise MindcraftUnavailable(
                    "No unique ready Mindcraft agent is available; configure agent_name."
                )
            if not status or not status.get("in_game") or not status.get("socket_connected"):
                raise MindcraftUnavailable(f"Mindcraft agent '{target}' is not ready.")

        return target

    def context_snapshot(self) -> dict[str, object]:
        with self._lock:
            target = self._resolve_agent_locked()
            status = self._status_for_locked(target)
            state = deepcopy(self._states.get(target)) if target else None
            recent_output = [
                output
                for output_agent, output in self._recent_output
                if target is None or output_agent == target
            ]
            snapshot: dict[str, object] = {
                "connection": "connected" if self._connected else "disconnected",
                "target_agent": target or self.agent_name,
                "target_status": deepcopy(status),
                "agents": deepcopy(self._agents),
            }

        if isinstance(state, Mapping):
            snapshot["world"] = {
                key: deepcopy(state[key])
                for key in (
                    "gameplay",
                    "action",
                    "surroundings",
                    "inventory",
                    "nearby",
                )
                if key in state
            }
        if recent_output:
            snapshot["recent_output"] = recent_output
        return snapshot

    def _connection_loop(self) -> None:
        while not self._stop_event.is_set():
            retry_delay = self.reconnect_delay_s
            try:
                self._socket.connect(self.url, wait_timeout=self.connect_timeout)
                if self._stop_event.is_set():
                    self._socket.disconnect()
                    break
                self._socket.wait()
            except Exception as exc:
                if not self._stop_event.is_set():
                    retry_delay = self._record_connection_failure(exc)
            finally:
                self._set_connected(False)
            self._stop_event.wait(retry_delay)

    def _record_connection_failure(self, exc: Exception) -> float:
        with self._lock:
            self._connection_failure_count += 1
            failure_count = self._connection_failure_count
            should_warn = not self._outage_warning_logged
            self._outage_warning_logged = True

        retry_delay = min(
            self.reconnect_max_delay_s,
            self.reconnect_delay_s * (2 ** min(failure_count - 1, 10)),
        )
        if should_warn:
            logger.warning(
                "Mindcraft unavailable at %s; retrying in the background",
                self.url,
            )
        logger.debug(
            "Mindcraft connection attempt %d failed; retrying in %.1fs: %s",
            failure_count,
            retry_delay,
            exc,
        )
        return retry_delay

    def _on_connect(self, *_args) -> None:
        with self._lock:
            recovered = self._outage_warning_logged
            self._connected = True
            self._connection_failure_count = 0
            self._outage_warning_logged = False
        if recovered:
            logger.info("Mindcraft connection restored at %s", self.url)
        else:
            logger.info("Connected to Mindcraft at %s", self.url)
        try:
            with self._emit_lock:
                self._socket.emit("listen-to-agents")
        except Exception:
            logger.exception("Failed to subscribe to Mindcraft state updates")

    def _on_disconnect(self, *_args) -> None:
        self._set_connected(False)
        logger.info("Disconnected from Mindcraft")

    def _on_agents_status(self, agents) -> None:
        if not isinstance(agents, list):
            logger.warning("Ignored malformed Mindcraft agents-status payload")
            return
        normalized = [dict(agent) for agent in agents if isinstance(agent, Mapping)]
        with self._lock:
            self._agents = normalized

    def _on_state_update(self, states) -> None:
        if not isinstance(states, Mapping):
            logger.warning("Ignored malformed Mindcraft state-update payload")
            return
        with self._lock:
            self._states = {str(name): deepcopy(state) for name, state in states.items()}

    def _on_bot_output(self, agent_name, message) -> None:
        with self._lock:
            self._recent_output.append((str(agent_name), deepcopy(message)))

    def _on_command_result(self, payload) -> None:
        if not isinstance(payload, Mapping):
            logger.warning("Ignored malformed Mindcraft command-result payload")
            return
        handler = self._command_result_handler
        if callable(handler):
            try:
                handler(dict(payload))
            except Exception:
                logger.exception("Mindcraft command-result handler failed")

    def _set_connected(self, connected: bool) -> None:
        with self._lock:
            self._connected = connected

    def _resolve_agent_locked(self) -> str | None:
        if self.agent_name:
            return self.agent_name
        ready = [
            str(agent.get("name"))
            for agent in self._agents
            if agent.get("name")
            and agent.get("in_game")
            and agent.get("socket_connected")
        ]
        return ready[0] if len(ready) == 1 else None

    def _status_for_locked(self, agent_name: str | None) -> dict[str, object] | None:
        if agent_name is None:
            return None
        return next(
            (agent for agent in self._agents if agent.get("name") == agent_name),
            None,
        )


class MindcraftIntegration:
    name = "mindcraft"

    def __init__(
        self,
        client: MindcraftClient,
        context_enabled: bool = True,
        events_enabled: bool = False,
        ambient_session_id: str | None = None,
    ):
        self.client = client
        self.context_enabled = context_enabled
        self.events_enabled = events_enabled
        self.ambient_session_id = ambient_session_id
        self._publisher: EventPublisher | None = None

    def registered_events(self) -> list[EventSpec]:
        if not self.events_enabled:
            return []
        allowed = tuple(tool.spec.capability for tool in self.registered_tools())
        schema = {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "minLength": 1},
                "agent_name": {"type": "string", "minLength": 1},
                "command": {"type": "string", "minLength": 1},
                "status": {"type": "string"},
                "message": {"type": "string"},
                "interrupted": {"type": "boolean"},
                "timed_out": {"type": "boolean"},
                "state": {"type": ["object", "null"]},
            },
            "required": [
                "request_id", "agent_name", "command", "status", "message",
                "interrupted", "timed_out",
            ],
            "additionalProperties": False,
        }
        return [
            EventSpec(
                event=EventId(self.name, "command_completed"),
                description="A correlated Mindcraft command completed successfully.",
                payload_schema=schema,
                allowed_capabilities=allowed,
                notification_policy=NotificationPolicy.MODEL_DECIDES,
                replay_policy=ReplayPolicy.NEVER,
                priority=50,
            ),
            EventSpec(
                event=EventId(self.name, "command_failed"),
                description="A correlated Mindcraft command failed, timed out, or was interrupted.",
                payload_schema=schema,
                allowed_capabilities=allowed,
                notification_policy=NotificationPolicy.MODEL_DECIDES,
                replay_policy=ReplayPolicy.NEVER,
                priority=25,
            ),
        ]

    def start(self, publisher: EventPublisher) -> None:
        self._publisher = publisher
        self.client.set_command_result_handler(self._publish_command_result)
        self.client.start()

    def _publish_command_result(self, payload: dict[str, object]) -> None:
        if not self.events_enabled or self._publisher is None:
            return
        request_id = str(payload.get("request_id", "")).strip()
        status = str(payload.get("status", "error")).lower()
        event_name = "command_completed" if status == "success" else "command_failed"
        normalized = {
            "request_id": request_id,
            "agent_name": str(payload.get("agent_name", self.client.agent_name or "unknown")),
            "command": str(payload.get("command", "")),
            "status": status,
            "message": str(payload.get("message", "")),
            "interrupted": bool(payload.get("interrupted", False)),
            "timed_out": bool(payload.get("timed_out", payload.get("timedout", False))),
            "state": payload.get("state") if isinstance(payload.get("state"), Mapping) else None,
        }
        self._publisher(IntegrationEvent(
            event=EventId(self.name, event_name),
            payload=normalized,
            session_id=self.ambient_session_id if not request_id else None,
            correlation_id=request_id or None,
            deduplication_key=request_id or None,
        ))

    def registered_tools(self) -> list[RegisteredTool]:
        available = self.client.is_available
        return [
            RegisteredTool(
                spec=ToolSpec(
                    capability=CapabilityId(self.name, "stop"),
                    description=(
                        "Stop the configured Minecraft agent's current action and any "
                        "continuous goal. Use this for an immediate stop request."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
                handler=self._stop,
                available=available,
            ),
            RegisteredTool(
                spec=ToolSpec(
                    capability=CapabilityId(self.name, "go_to_player"),
                    description=(
                        "Move the configured Minecraft agent near a player. Use this "
                        "direct action instead of delegating the request to Mindcraft's model."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "player_name": self._player_name_schema(),
                            "closeness": {
                                "type": "number",
                                "description": "Desired distance from the player in blocks.",
                                "minimum": 0,
                                "maximum": 128,
                            },
                        },
                        "required": ["player_name", "closeness"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._go_to_player,
                available=available,
            ),
            RegisteredTool(
                spec=ToolSpec(
                    capability=CapabilityId(self.name, "follow_player"),
                    description=(
                        "Continuously follow a player at a specified distance. Use "
                        "mindcraft__stop to stop following."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "player_name": self._player_name_schema(),
                            "follow_distance": {
                                "type": "number",
                                "description": "Following distance in blocks.",
                                "minimum": 0,
                                "maximum": 128,
                            },
                        },
                        "required": ["player_name", "follow_distance"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._follow_player,
                available=available,
            ),
            RegisteredTool(
                spec=ToolSpec(
                    capability=CapabilityId(self.name, "collect_blocks"),
                    description=(
                        "Collect a bounded number of nearby blocks of one Minecraft block "
                        "type without invoking Mindcraft's planning model."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "block_type": {
                                "type": "string",
                                "description": "Minecraft block identifier, such as oak_log.",
                                "pattern": "^[a-z0-9_]+$",
                                "minLength": 1,
                                "maxLength": 64,
                            },
                            "count": {
                                "type": "integer",
                                "description": "Number of blocks to collect.",
                                "minimum": 1,
                                "maximum": 256,
                            },
                        },
                        "required": ["block_type", "count"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._collect_blocks,
                available=available,
            ),
            RegisteredTool(
                spec=ToolSpec(
                    capability=CapabilityId(self.name, "send_message"),
                    description=(
                        "Delegate a complex, open-ended objective or conversational message to "
                        "the configured Mindcraft agent and its planning model. Prefer a typed "
                        "Mindcraft action when one directly matches the request."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The instruction or message for the Minecraft agent.",
                                "minLength": 1,
                                "maxLength": 4000,
                            },
                        },
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._send_message,
                available=available,
            ),
        ]

    @staticmethod
    def _player_name_schema() -> dict[str, object]:
        return {
            "type": "string",
            "description": "Exact Minecraft player name.",
            "pattern": "^[A-Za-z0-9_]+$",
            "minLength": 1,
            "maxLength": 16,
        }

    def _stop(
        self,
        _arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        return self._send_direct_command("!endGoal", "Stop", context)

    def _go_to_player(
        self,
        arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        command = (
            f'!goToPlayer("{arguments["player_name"]}", '
            f'{json.dumps(arguments["closeness"])})'
        )
        return self._send_direct_command(command, "Go-to-player", context)

    def _follow_player(
        self,
        arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        command = (
            f'!followPlayer("{arguments["player_name"]}", '
            f'{json.dumps(arguments["follow_distance"])})'
        )
        return self._send_direct_command(command, "Follow-player", context)

    def _collect_blocks(
        self,
        arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        command = f'!collectBlocks("{arguments["block_type"]}", {arguments["count"]})'
        return self._send_direct_command(command, "Collect-blocks", context)

    def _send_direct_command(
        self,
        command: str,
        label: str,
        context: InvocationContext,
    ) -> ToolResult:
        if self.events_enabled and not context.invocation_id:
            return ToolResult.error("Mindcraft command execution requires an invocation ID.")
        try:
            target = self.client.send_command(command, context.invocation_id if self.events_enabled else None)
        except MindcraftUnavailable as exc:
            return ToolResult.unavailable(str(exc))
        content = f"{label} command accepted by Mindcraft agent '{target}'. Execution is asynchronous."
        if self.events_enabled:
            return ToolResult.pending(content, context.invocation_id or "")
        return ToolResult.success(content)

    def _send_message(
        self,
        arguments: Mapping[str, object],
        _context: InvocationContext,
    ) -> ToolResult:
        try:
            target = self.client.send_message(str(arguments["message"]))
        except MindcraftUnavailable as exc:
            return ToolResult.unavailable(str(exc))
        return ToolResult.success(
            f"Instruction accepted by Mindcraft agent '{target}'. Execution is asynchronous."
        )

    def context(self, _invocation: InvocationContext) -> ContextContribution | None:
        if not self.context_enabled:
            return None
        content = json.dumps(
            self.client.context_snapshot(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return ContextContribution(
            source=self.name,
            content=(
                "Observed Mindcraft state (cached, may be stale, and is not tool output):\n"
                f"{content}"
            ),
        )

    def close(self) -> None:
        self.client.close()
