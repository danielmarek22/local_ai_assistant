from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from app.integrations.contracts import (
    CapabilityId,
    ContextContribution,
    Integration,
    InvocationContext,
    RegisteredTool,
    ToolCall,
    ToolResult,
)


logger = logging.getLogger("integration_registry")


class IntegrationRegistry:
    def __init__(self, integrations: Iterable[Integration] = ()):
        self._integrations: dict[str, Integration] = {}
        self._tools: dict[CapabilityId, RegisteredTool] = {}

        for integration in integrations:
            self.register(integration)

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

    def get_native_tools(self) -> list[dict]:
        native_tools = []
        for capability, registered in sorted(self._tools.items()):
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

    @staticmethod
    def _is_available(registered: RegisteredTool) -> bool:
        try:
            return registered.is_available()
        except Exception:
            logger.exception("Availability check failed for %s", registered.spec.capability)
            return False
