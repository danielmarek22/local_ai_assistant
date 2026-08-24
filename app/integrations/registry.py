from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from app.integrations.contracts import (
    CapabilityId,
    ContextContribution,
    EventId,
    EventPublisher,
    EventSpec,
    Integration,
    IntegrationEvent,
    InvocationContext,
    RegisteredTool,
    ToolCall,
    ToolResult,
    NotificationPolicy,
    ReplayPolicy,
    EventAttachmentRef,
)


logger = logging.getLogger("integration_registry")


class IntegrationRegistry:
    def __init__(self, integrations: Iterable[Integration] = ()):
        self._integrations: dict[str, Integration] = {}
        self._tools: dict[CapabilityId, RegisteredTool] = {}
        self._events: dict[EventId, EventSpec] = {}
        self._started = False

        for integration in integrations:
            self.register(integration)
        self._validate_event_allowlists()

    def register(self, integration: Integration) -> None:
        name = getattr(integration, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Integration must declare a name")
        CapabilityId(name, "validation")
        if name in self._integrations:
            raise ValueError(f"Duplicate integration: {name}")

        tools = integration.registered_tools()
        pending: list[tuple[CapabilityId, RegisteredTool]] = []
        for registered in tools:
            if not isinstance(registered, RegisteredTool):
                raise ValueError(f"Integration {name!r} returned an invalid registered tool")
            capability = registered.spec.capability
            if capability.integration != name:
                raise ValueError(
                    f"Capability {capability} does not belong to integration {name!r}"
                )
            if capability in self._tools or any(item[0] == capability for item in pending):
                raise ValueError(f"Duplicate capability: {capability}")
            if not callable(registered.handler):
                raise ValueError(f"Capability {capability} has no callable handler")
            if not isinstance(registered.spec.description, str) or not registered.spec.description.strip():
                raise ValueError(f"Capability {capability} must declare a description")
            schema = registered.spec.input_schema
            if not isinstance(schema, Mapping) or schema.get("type") != "object":
                raise ValueError(f"Capability {capability} must use an object input schema")
            try:
                Draft202012Validator.check_schema(dict(schema))
            except SchemaError as exc:
                raise ValueError(f"Capability {capability} has an invalid input schema") from exc
            pending.append((capability, registered))

        self._integrations[name] = integration
        self._tools.update(pending)

        event_provider = getattr(integration, "registered_events", None)
        if callable(event_provider):
            for spec in event_provider():
                self._register_event(name, spec)

    def get_native_tools(
        self,
        allowed_capabilities: set[CapabilityId] | None = None,
    ) -> list[dict]:
        native_tools = []
        for capability, registered in sorted(self._tools.items()):
            if allowed_capabilities is not None and capability not in allowed_capabilities:
                continue
            if not self._is_available(registered):
                continue
            native_tools.append({
                "type": "function",
                "function": {
                    "name": str(capability),
                    "description": registered.spec.description,
                    "parameters": dict(registered.spec.input_schema),
                },
            })
        return native_tools

    def get_event_spec(self, event: EventId) -> EventSpec | None:
        return self._events.get(event)

    def get_integration(self, name: str) -> Integration | None:
        return self._integrations.get(name)

    def validate_event(self, event: IntegrationEvent) -> EventSpec:
        if not isinstance(event, IntegrationEvent):
            raise ValueError("Invalid integration event")
        spec = self._events.get(event.event)
        if spec is None:
            raise ValueError(f"Unknown integration event: {event.event}")
        try:
            uuid.UUID(event.event_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integration event UUID: {event.event_id!r}") from exc
        if event.session_id is not None and (
            not isinstance(event.session_id, str) or not event.session_id.strip()
        ):
            raise ValueError("Event session_id must be a non-empty string when provided")
        if not isinstance(event.occurred_at, datetime) or event.occurred_at.tzinfo is None:
            raise ValueError("Event occurred_at must be a timezone-aware datetime")
        for label, value in (
            ("correlation_id", event.correlation_id),
            ("causation_id", event.causation_id),
            ("root_event_id", event.root_event_id),
            ("deduplication_key", event.deduplication_key),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Event {label} must be a non-empty string when provided")
        if any(not isinstance(item, EventAttachmentRef) for item in event.attachments):
            raise ValueError("Event attachments must use EventAttachmentRef")
        if not isinstance(event.payload, Mapping):
            raise ValueError(f"Invalid payload for {event.event}: expected an object")
        try:
            Draft202012Validator(dict(spec.payload_schema)).validate(dict(event.payload))
        except ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path)
            location = f" at {path}" if path else ""
            raise ValueError(
                f"Invalid payload for {event.event}{location}: {exc.message}"
            ) from exc
        if event.root_event_id and not event.causation_id:
            raise ValueError("Event root_event_id requires causation_id")
        return spec

    def start(self, publisher: EventPublisher) -> None:
        if self._started:
            return
        self._validate_event_allowlists()
        started: list[Integration] = []
        try:
            for integration in self._integrations.values():
                start = getattr(integration, "start", None)
                if callable(start):
                    start(publisher)
                started.append(integration)
        except Exception:
            logger.exception("Integration startup failed")
            for integration in reversed(started):
                close = getattr(integration, "close", None)
                if callable(close):
                    close()
            raise
        self._started = True

    def invoke(self, call: ToolCall, context: InvocationContext) -> ToolResult:
        registered = self._tools.get(call.capability)
        if registered is None:
            return ToolResult.error(f"Unknown capability: {call.capability}")
        if not self._is_available(registered):
            return ToolResult.unavailable(f"Capability is unavailable: {call.capability}")
        if not isinstance(call.arguments, Mapping):
            return ToolResult.error(f"Invalid arguments for {call.capability}: expected an object")

        try:
            Draft202012Validator(dict(registered.spec.input_schema)).validate(dict(call.arguments))
        except ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path)
            location = f" at {path}" if path else ""
            return ToolResult.error(
                f"Invalid arguments for {call.capability}{location}: {exc.message}"
            )

        try:
            result = registered.handler(dict(call.arguments), context)
        except Exception:
            logger.exception("Capability %s failed", call.capability)
            return ToolResult.error(f"Capability execution failed: {call.capability}")

        if not isinstance(result, ToolResult):
            logger.error("Capability %s returned an invalid result", call.capability)
            return ToolResult.error(f"Capability returned an invalid result: {call.capability}")
        return result

    def collect_context(self, invocation: InvocationContext, max_chars: int) -> str | None:
        remaining = max(0, int(max_chars))
        sections: list[str] = []
        for name, integration in sorted(self._integrations.items()):
            if remaining <= 0:
                break
            provider = getattr(integration, "context", None)
            if not callable(provider):
                continue
            try:
                contribution = provider(invocation)
            except Exception:
                logger.exception("Integration %s context provider failed", name)
                continue
            if contribution is None:
                continue
            if not isinstance(contribution, ContextContribution):
                logger.warning("Integration %s returned invalid context", name)
                continue
            if contribution.source != name:
                logger.warning(
                    "Integration %s returned context for unexpected source %s",
                    name,
                    contribution.source,
                )
                continue
            content = contribution.content.strip()
            if not content:
                continue
            bounded = content[:remaining]
            sections.append(f"--- {contribution.source} ---\n{bounded}")
            remaining -= len(bounded)
        return "\n\n".join(sections) or None

    def close(self) -> None:
        for name, integration in reversed(self._integrations.items()):
            close = getattr(integration, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception:
                logger.exception("Integration %s failed during shutdown", name)
        self._started = False

    def _register_event(self, integration_name: str, spec: EventSpec) -> None:
        if not isinstance(spec, EventSpec):
            raise ValueError(f"Integration {integration_name!r} returned an invalid event spec")
        if spec.event.integration != integration_name:
            raise ValueError(
                f"Event {spec.event} does not belong to integration {integration_name!r}"
            )
        if spec.event in self._events:
            raise ValueError(f"Duplicate event: {spec.event}")
        if not isinstance(spec.description, str) or not spec.description.strip():
            raise ValueError(f"Event {spec.event} must declare a description")
        if not isinstance(spec.payload_schema, Mapping) or spec.payload_schema.get("type") != "object":
            raise ValueError(f"Event {spec.event} must use an object payload schema")
        try:
            Draft202012Validator.check_schema(dict(spec.payload_schema))
        except SchemaError as exc:
            raise ValueError(f"Event {spec.event} has an invalid payload schema") from exc
        if not isinstance(spec.priority, int) or spec.priority < 0:
            raise ValueError(f"Event {spec.event} has an invalid priority")
        if spec.coalesce_window_s < 0:
            raise ValueError(f"Event {spec.event} has an invalid coalescing window")
        if len(set(spec.allowed_capabilities)) != len(spec.allowed_capabilities):
            raise ValueError(f"Event {spec.event} has duplicate allowed capabilities")
        if any(not isinstance(item, CapabilityId) for item in spec.allowed_capabilities):
            raise ValueError(f"Event {spec.event} has an invalid allowed capability")
        if not isinstance(spec.notification_policy, NotificationPolicy):
            raise ValueError(f"Event {spec.event} has an invalid notification policy")
        if not isinstance(spec.replay_policy, ReplayPolicy):
            raise ValueError(f"Event {spec.event} has an invalid replay policy")
        self._events[spec.event] = spec

    def _validate_event_allowlists(self) -> None:
        for spec in self._events.values():
            unknown = [capability for capability in spec.allowed_capabilities if capability not in self._tools]
            if unknown:
                names = ", ".join(str(capability) for capability in unknown)
                raise ValueError(f"Event {spec.event} allows unknown capabilities: {names}")

    @staticmethod
    def _is_available(registered: RegisteredTool) -> bool:
        try:
            return registered.is_available()
        except Exception:
            logger.exception("Availability check failed for %s", registered.spec.capability)
            return False
