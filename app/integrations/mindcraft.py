from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from app.integrations.contracts import (
    ContextContribution,
    EventAttachmentRef,
    InvocationContext,
    RegisteredTool,
    ToolResult,
    EventId,
    EventPublisher,
    EventSpec,
    IntegrationEvent,
    NotificationPolicy,
    ReplayPolicy,
)
from app.integrations.mindcraft_capabilities import build_mindcraft_tool_specs


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
        self._operation_event_handler = None
        self._agent_event_handler = None
        self._protocol_ready_handler = None
        self._protocol_manifest: dict[str, object] | None = None
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
        self._socket.on("operation-event", self._on_operation_event)
        self._socket.on("agent-event", self._on_agent_event)

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

    def send_action(
        self,
        action: str,
        arguments: Mapping[str, object],
        operation_id: str,
    ) -> str:
        target = self._ready_target()
        if not self.supports("typed_actions"):
            raise MindcraftUnavailable(
                "Mindcraft does not advertise the typed action protocol; restart the updated fork."
            )
        payload = {
            "protocol_version": 1,
            "operation_id": operation_id,
            "from": "local_assistant",
            "action": action,
            "arguments": dict(arguments),
        }
        try:
            with self._emit_lock:
                self._socket.emit("execute-action", (target, payload))
        except Exception as exc:
            raise MindcraftUnavailable(
                f"Could not dispatch action to Mindcraft agent '{target}'."
            ) from exc
        return target

    def supports(self, feature: str) -> bool:
        with self._lock:
            manifest = deepcopy(self._protocol_manifest)
        return bool(
            isinstance(manifest, Mapping)
            and feature in manifest.get("features", [])
        )

    def controller_mode(self) -> str:
        with self._lock:
            target = self._resolve_agent_locked()
            manifest = deepcopy(self._protocol_manifest)
        if isinstance(manifest, Mapping):
            modes = manifest.get("agent_controller_modes")
            if isinstance(modes, Mapping) and target in modes:
                return str(modes[target])
            return str(manifest.get("controller_mode", "hybrid"))
        return "hybrid"

    def set_command_result_handler(self, handler) -> None:
        self._command_result_handler = handler

    def set_operation_event_handler(self, handler) -> None:
        self._operation_event_handler = handler

    def set_agent_event_handler(self, handler) -> None:
        self._agent_event_handler = handler

    def set_protocol_ready_handler(self, handler) -> None:
        self._protocol_ready_handler = handler

    def query_operation(self, operation_id: str) -> dict[str, object] | None:
        target = self._ready_target()
        try:
            with self._emit_lock:
                result = self._socket.call(
                    "get-operation",
                    (target, operation_id),
                    timeout=self.connect_timeout,
                )
        except Exception as exc:
            raise MindcraftUnavailable(
                f"Could not query operation {operation_id} from Mindcraft agent '{target}'."
            ) from exc
        return dict(result) if isinstance(result, Mapping) else None

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
                "protocol": deepcopy(self._protocol_manifest),
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
                self._socket.emit(
                    "integration-hello",
                    {"protocol_version": 1, "client": "local_ai_assistant"},
                    callback=self._on_protocol_manifest,
                )
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

    def _on_protocol_manifest(self, payload) -> None:
        if not isinstance(payload, Mapping):
            logger.warning("Ignored malformed Mindcraft integration manifest")
            return
        with self._lock:
            self._protocol_manifest = dict(payload)
        logger.info(
            "Mindcraft integration protocol ready (version=%s, features=%s)",
            payload.get("protocol_version"),
            len(payload.get("features", [])),
        )
        handler = self._protocol_ready_handler
        if callable(handler):
            threading.Thread(
                target=handler,
                name="mindcraft-operation-recovery",
                daemon=True,
            ).start()

    def _on_operation_event(self, payload) -> None:
        if not isinstance(payload, Mapping):
            logger.warning("Ignored malformed Mindcraft operation event")
            return
        handler = self._operation_event_handler
        if callable(handler):
            try:
                handler(dict(payload))
            except Exception:
                logger.exception("Mindcraft operation-event handler failed")

    def _on_agent_event(self, payload) -> None:
        if not isinstance(payload, Mapping):
            logger.warning("Ignored malformed Mindcraft agent event")
            return
        handler = self._agent_event_handler
        if callable(handler):
            try:
                handler(dict(payload))
            except Exception:
                logger.exception("Mindcraft agent-event handler failed")

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
        autonomous_events: tuple[str, ...] = (
            "critical_health",
            "died",
            "disconnected",
        ),
        attachment_dir: str | Path = "static/uploads/events/mindcraft",
        operation_store=None,
    ):
        self.client = client
        self.context_enabled = context_enabled
        self.events_enabled = events_enabled
        self.ambient_session_id = ambient_session_id
        self.autonomous_events = frozenset(autonomous_events)
        self.attachment_dir = Path(attachment_dir)
        self.operation_store = operation_store
        self._publisher: EventPublisher | None = None
        self._bound_session_id = ambient_session_id

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
        specs = [
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
        agent_event_schema = {
            "type": "object",
            "properties": {
                "protocol_version": {"type": "integer"},
                "source_event_id": {"type": "string", "minLength": 1},
                "sequence": {"type": "integer", "minimum": 1},
                "agent_name": {"type": "string", "minLength": 1},
                "occurred_at": {"type": "string", "minLength": 1},
                "details": {"type": "object"},
                "state": {"type": ["object", "null"]},
            },
            "required": [
                "protocol_version", "source_event_id", "sequence", "agent_name",
                "occurred_at", "details", "state",
            ],
            "additionalProperties": False,
        }
        event_policies = {
            "spawned": (NotificationPolicy.MODEL_DECIDES, 70, 0.0),
            "disconnected": (NotificationPolicy.ALWAYS_NOTIFY, 10, 0.0),
            "damage_taken": (NotificationPolicy.NEVER_NOTIFY, 45, 2.0),
            "critical_health": (NotificationPolicy.MODEL_DECIDES, 5, 3.0),
            "died": (NotificationPolicy.ALWAYS_NOTIFY, 5, 0.0),
            "respawned": (NotificationPolicy.MODEL_DECIDES, 20, 0.0),
            "player_joined": (NotificationPolicy.NEVER_NOTIFY, 90, 2.0),
            "player_left": (NotificationPolicy.NEVER_NOTIFY, 90, 2.0),
            "player_spoke": (NotificationPolicy.MODEL_DECIDES, 100, 1.0),
            "delegation_rejected": (NotificationPolicy.NEVER_NOTIFY, 100, 0.0),
        }
        tool_capabilities = {
            tool.spec.capability.action: tool.spec.capability
            for tool in self.registered_tools()
        }
        event_allowlists = {
            "spawned": ("capture_view", "say"),
            "disconnected": (),
            "damage_taken": ("stop", "capture_view"),
            "critical_health": ("stop", "capture_view", "say"),
            "died": ("stop", "capture_view", "say"),
            "respawned": ("capture_view", "say"),
            "player_joined": ("say",),
            "player_left": (),
            "player_spoke": ("say",),
            "delegation_rejected": (),
        }
        for event_name, (notification, priority, coalesce) in event_policies.items():
            specs.append(EventSpec(
                event=EventId(self.name, event_name),
                description=f"Mindcraft reported the exact game event '{event_name}'.",
                payload_schema=agent_event_schema,
                allowed_capabilities=tuple(
                    tool_capabilities[action]
                    for action in event_allowlists[event_name]
                ),
                notification_policy=notification,
                replay_policy=ReplayPolicy.NEVER,
                priority=priority,
                coalesce_window_s=coalesce,
            ))
        return specs

    def start(self, publisher: EventPublisher) -> None:
        self._publisher = publisher
        self.client.set_command_result_handler(self._publish_command_result)
        self.client.set_operation_event_handler(self._publish_operation_event)
        self.client.set_agent_event_handler(self._publish_agent_event)
        self.client.set_protocol_ready_handler(self._reconcile_pending_operations)
        self.client.start()

    def _reconcile_pending_operations(self) -> None:
        if self.operation_store is None or self._publisher is None:
            return
        for operation in self.operation_store.pending_operations("mindcraft__"):
            try:
                payload = self.client.query_operation(operation.invocation_id)
            except MindcraftUnavailable:
                logger.debug("Mindcraft operation recovery paused while unavailable")
                return
            if payload is None:
                payload = {
                    "operation_id": operation.invocation_id,
                    "agent_name": self.client.agent_name or "unknown",
                    "action": operation.capability.removeprefix("mindcraft__"),
                    "status": "failed",
                    "code": "operation_lost",
                    "message": "Mindcraft no longer has a record of this pending operation.",
                    "terminal": True,
                    "state": None,
                }
            if bool(payload.get("terminal", False)):
                self._publish_operation_event(payload)

    def _publish_command_result(self, payload: dict[str, object]) -> None:
        if not self.events_enabled or self._publisher is None:
            return
        request_id = str(payload.get("request_id", "")).strip()
        status = str(payload.get("status", "error")).lower()
        event_name = "command_completed" if status == "success" else "command_failed"
        if (
            self.operation_store is not None
            and request_id
            and self.operation_store.has_event_deduplication_key(
                f"mindcraft__{event_name}", request_id
            )
        ):
            return
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
        if self.operation_store is not None and request_id:
            self.operation_store.finish_operation(
                request_id,
                "success" if event_name == "command_completed" else "error",
                normalized["message"],
            )
        self._publisher(IntegrationEvent(
            event=EventId(self.name, event_name),
            payload=normalized,
            session_id=self.ambient_session_id if not request_id else None,
            correlation_id=request_id or None,
            deduplication_key=request_id or None,
        ))

    def _publish_operation_event(self, payload: dict[str, object]) -> None:
        if not self.events_enabled or self._publisher is None:
            return
        status = str(payload.get("status", "failed")).lower()
        if not bool(payload.get("terminal", False)) and status not in {
            "completed", "failed", "cancelled", "timed_out", "unavailable"
        }:
            return
        operation_id = str(payload.get("operation_id", "")).strip()
        event_name = "command_completed" if status == "completed" else "command_failed"
        if (
            self.operation_store is not None
            and operation_id
            and self.operation_store.has_event_deduplication_key(
                f"mindcraft__{event_name}", operation_id
            )
        ):
            return
        data = payload.get("data") if isinstance(payload.get("data"), Mapping) else None
        attachment = data.get("attachment") if isinstance(data, Mapping) else None
        attachments: tuple[EventAttachmentRef, ...] = ()
        if isinstance(attachment, Mapping):
            persisted = self._persist_attachment(attachment)
            if persisted is not None:
                attachments = (persisted,)
            data = {key: value for key, value in data.items() if key != "attachment"}
        normalized = {
            "request_id": operation_id,
            "agent_name": str(payload.get("agent_name", self.client.agent_name or "unknown")),
            "command": str(payload.get("action", "")),
            "status": status,
            "message": str(payload.get("message", "")),
            "interrupted": bool(payload.get("interrupted", status == "cancelled")),
            "timed_out": bool(payload.get("timed_out", status == "timed_out")),
            "state": payload.get("state") if isinstance(payload.get("state"), Mapping) else None,
        }
        if data:
            normalized["state"] = {
                **(normalized["state"] or {}),
                "operation_data": data,
            }
        if self.operation_store is not None and operation_id:
            self.operation_store.finish_operation(
                operation_id,
                "success" if event_name == "command_completed" else "error",
                normalized["message"],
            )
        self._publisher(IntegrationEvent(
            event=EventId(self.name, event_name),
            payload=normalized,
            correlation_id=operation_id or None,
            deduplication_key=operation_id or None,
            attachments=attachments,
        ))

    def _publish_agent_event(self, payload: dict[str, object]) -> None:
        if not self.events_enabled or self._publisher is None:
            return
        event_name = str(payload.get("event", "")).strip()
        if event_name not in {
            "spawned", "disconnected", "damage_taken", "critical_health", "died",
            "respawned", "player_joined", "player_left", "player_spoke",
            "delegation_rejected",
        }:
            logger.warning("Ignored unknown Mindcraft agent event %r", event_name)
            return
        source_event_id = str(payload.get("event_id", "")).strip()
        occurred_at_raw = str(payload.get("occurred_at", "")).strip()
        try:
            occurred_at = datetime.fromisoformat(occurred_at_raw.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Ignored Mindcraft event with invalid timestamp")
            return
        normalized = {
            "protocol_version": int(payload.get("protocol_version", 1)),
            "source_event_id": source_event_id,
            "sequence": int(payload.get("sequence", 0)),
            "agent_name": str(payload.get("agent_name", self.client.agent_name or "unknown")),
            "occurred_at": occurred_at_raw,
            "details": dict(payload.get("payload", {})) if isinstance(payload.get("payload"), Mapping) else {},
            "state": payload.get("state") if isinstance(payload.get("state"), Mapping) else None,
        }
        target_session = (
            self._bound_session_id
            if event_name in self.autonomous_events
            else None
        )
        deduplication_key = source_event_id or f"{normalized['agent_name']}:{normalized['sequence']}"
        if event_name in {"damage_taken", "critical_health"}:
            deduplication_key = f"{normalized['agent_name']}:{event_name}"
        elif event_name in {"player_joined", "player_left"}:
            deduplication_key = (
                f"{normalized['agent_name']}:{event_name}:"
                f"{normalized['details'].get('player_name', 'unknown')}"
            )
        self._publisher(IntegrationEvent(
            event=EventId(self.name, event_name),
            payload=normalized,
            session_id=target_session,
            occurred_at=occurred_at,
            deduplication_key=deduplication_key,
        ))

    def _persist_attachment(
        self,
        attachment: Mapping[str, object],
    ) -> EventAttachmentRef | None:
        raw = attachment.get("data_base64")
        if not isinstance(raw, str):
            return None
        try:
            payload = base64.b64decode(raw, validate=True)
        except ValueError:
            logger.warning("Ignored invalid Mindcraft vision attachment")
            return None
        if not payload or len(payload) > 5 * 1024 * 1024:
            logger.warning("Ignored empty or oversized Mindcraft vision attachment")
            return None
        digest = hashlib.sha256(payload).hexdigest()
        advertised = str(attachment.get("sha256", ""))
        if advertised and advertised != digest:
            logger.warning("Ignored Mindcraft vision attachment with mismatched digest")
            return None
        directory = self.attachment_dir / digest[:2] / digest
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest}.jpg"
        path.write_bytes(payload)
        return EventAttachmentRef(
            name=str(attachment.get("name", "mindcraft-view.jpg")),
            mime_type="image/jpeg",
            storage_path=str(path),
            sha256=digest,
            size_bytes=len(payload),
        )

    def registered_tools(self) -> list[RegisteredTool]:
        available = self.client.is_available
        handlers = {
            "stop": self._stop,
            "say": self._say,
            "go_to_player": self._go_to_player,
            "follow_player": self._follow_player,
            "collect_blocks": self._collect_blocks,
            "collect_resource": self._collect_resource,
            "chop_tree": self._chop_tree,
            "capture_view": self._capture_view,
            "look_at_position": self._look_at_position,
            "look_at_player": self._look_at_player,
            "send_message": self._send_message,
        }
        return [
            RegisteredTool(
                spec=spec,
                handler=handlers[spec.capability.action],
                available=(
                    (lambda: available() and self.client.controller_mode() != "external")
                    if spec.capability.action == "send_message"
                    else available
                ),
            )
            for spec in build_mindcraft_tool_specs(self.name)
        ]

    def _stop(
        self,
        _arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        return self._send_action("stop", {}, "Stop", context, "!endGoal")

    def _say(self, arguments: Mapping[str, object], context: InvocationContext) -> ToolResult:
        return self._send_action("say", arguments, "Minecraft-chat", context)

    def _go_to_player(
        self,
        arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        command = (
            f'!goToPlayer("{arguments["player_name"]}", '
            f'{json.dumps(arguments["closeness"])})'
        )
        return self._send_action("go_to_player", arguments, "Go-to-player", context, command)

    def _follow_player(
        self,
        arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        command = (
            f'!followPlayer("{arguments["player_name"]}", '
            f'{json.dumps(arguments["follow_distance"])})'
        )
        return self._send_action("follow_player", arguments, "Follow-player", context, command)

    def _collect_blocks(
        self,
        arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        command = f'!collectBlocks("{arguments["block_type"]}", {arguments["count"]})'
        return self._send_action("collect_blocks", arguments, "Collect-blocks", context, command)

    def _collect_resource(self, arguments: Mapping[str, object], context: InvocationContext) -> ToolResult:
        return self._send_action("collect_resource", arguments, "Collect-resource", context)

    def _chop_tree(self, arguments: Mapping[str, object], context: InvocationContext) -> ToolResult:
        return self._send_action("chop_tree", arguments, "Chop-tree", context)

    def _capture_view(self, arguments: Mapping[str, object], context: InvocationContext) -> ToolResult:
        return self._send_action("capture_view", arguments, "Capture-view", context)

    def _look_at_position(self, arguments: Mapping[str, object], context: InvocationContext) -> ToolResult:
        command = f'!lookAtPosition({arguments["x"]}, {arguments["y"]}, {arguments["z"]})'
        return self._send_action("look_at_position", arguments, "Look-at-position", context, command)

    def _look_at_player(self, arguments: Mapping[str, object], context: InvocationContext) -> ToolResult:
        command = (
            f'!lookAtPlayer("{arguments["player_name"]}", '
            f'"{arguments["direction"]}")'
        )
        return self._send_action("look_at_player", arguments, "Look-at-player", context, command)

    def _send_action(
        self,
        action: str,
        arguments: Mapping[str, object],
        label: str,
        context: InvocationContext,
        legacy_command: str | None = None,
    ) -> ToolResult:
        if self.events_enabled and self.client.supports("typed_actions"):
            if not context.invocation_id:
                return ToolResult.error("Mindcraft action execution requires an invocation ID.")
            self._bound_session_id = context.session_id
            try:
                target = self.client.send_action(action, arguments, context.invocation_id)
            except MindcraftUnavailable as exc:
                return ToolResult.unavailable(str(exc))
            return ToolResult.pending(
                f"{label} operation accepted by Mindcraft agent '{target}'.",
                context.invocation_id,
            )
        if legacy_command is None:
            return ToolResult.unavailable(
                f"{label} requires the Mindcraft typed action protocol."
            )
        return self._send_direct_command(legacy_command, label, context)

    def _send_direct_command(
        self,
        command: str,
        label: str,
        context: InvocationContext,
    ) -> ToolResult:
        if self.events_enabled and not context.invocation_id:
            return ToolResult.error("Mindcraft command execution requires an invocation ID.")
        self._bound_session_id = context.session_id
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
        context: InvocationContext,
    ) -> ToolResult:
        try:
            target = self.client.send_message(str(arguments["message"]))
        except MindcraftUnavailable as exc:
            return ToolResult.unavailable(str(exc))
        self._bound_session_id = context.session_id
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
