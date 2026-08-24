import unittest

from app.integrations import (
    CapabilityId,
    ContextContribution,
    IntegrationRegistry,
    InvocationContext,
    RegisteredTool,
    ToolCall,
    ToolResult,
    ToolSpec,
    EventId,
    EventSpec,
    IntegrationEvent,
)


class FakeIntegration:
    def __init__(self, name="demo", context="", available=True, raises=False):
        self.name = name
        self.context_text = context
        self.calls = []
        self.available = available
        self.raises = raises
        self.closed = False

    def registered_tools(self):
        return [RegisteredTool(
            spec=ToolSpec(
                capability=CapabilityId(self.name, "run"),
                description="Run demo.",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            ),
            handler=self._run,
            available=lambda: self.available,
        )]

    def _run(self, arguments, _context):
        self.calls.append(dict(arguments))
        if self.raises:
            raise RuntimeError("boom")
        return ToolResult.success("done")

    def context(self, _invocation):
        if not self.context_text:
            return None
        return ContextContribution(self.name, self.context_text)

    def close(self):
        self.closed = True


class IntegrationRegistryTests(unittest.TestCase):
    def test_event_ids_and_payloads_are_strictly_validated(self):
        integration = FakeIntegration()
        integration.registered_events = lambda: [EventSpec(
            event=EventId("demo", "finished"),
            description="Finished.",
            payload_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            allowed_capabilities=(CapabilityId("demo", "run"),),
        )]
        registry = IntegrationRegistry([integration])

        spec = registry.validate_event(IntegrationEvent(
            EventId.parse("demo__finished"), {"ok": True}, "session-1"
        ))
        self.assertEqual(str(spec.event), "demo__finished")
        with self.assertRaises(ValueError):
            registry.validate_event(IntegrationEvent(spec.event, {"ok": "yes"}, "session-1"))
        with self.assertRaises(ValueError):
            registry.validate_event(IntegrationEvent(
                spec.event,
                {"ok": True},
                "session-1",
                root_event_id="root-without-cause",
            ))

    def test_event_unknown_allowlist_is_rejected(self):
        integration = FakeIntegration()
        integration.registered_events = lambda: [EventSpec(
            event=EventId("demo", "finished"),
            description="Finished.",
            payload_schema={"type": "object", "properties": {}},
            allowed_capabilities=(CapabilityId("missing", "run"),),
        )]
        with self.assertRaises(ValueError):
            IntegrationRegistry([integration])

    def test_close_calls_integration_cleanup(self):
        integration = FakeIntegration()
        registry = IntegrationRegistry([integration])

        registry.close()

        self.assertTrue(integration.closed)

    def test_cleanup_failure_does_not_stop_other_integrations(self):
        healthy = FakeIntegration(name="healthy")
        failing = FakeIntegration(name="failing")
        failing.close = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        registry = IntegrationRegistry([healthy, failing])

        with self.assertLogs("integration_registry", level="ERROR"):
            registry.close()

        self.assertTrue(healthy.closed)

    def test_capability_ids_require_namespaced_safe_format(self):
        self.assertEqual(str(CapabilityId.parse("demo__run")), "demo__run")
        with self.assertRaises(ValueError):
            CapabilityId.parse("demo.run")
        with self.assertRaises(ValueError):
            CapabilityId("demo", "bad__action")

    def test_duplicate_capabilities_are_rejected_at_registration(self):
        with self.assertRaises(ValueError):
            IntegrationRegistry([FakeIntegration(), FakeIntegration()])

    def test_schema_rejection_prevents_handler_execution(self):
        integration = FakeIntegration()
        registry = IntegrationRegistry([integration])

        result = registry.invoke(
            ToolCall(CapabilityId("demo", "run"), {"extra": "x"}),
            InvocationContext("session-1", "hello"),
        )

        self.assertEqual(result.status.value, "error")
        self.assertEqual(integration.calls, [])

    def test_native_tools_include_only_registered_available_capabilities(self):
        registry = IntegrationRegistry([FakeIntegration()])
        schemas = registry.get_native_tools()

        self.assertEqual(schemas[0]["function"]["name"], "demo__run")

        unavailable = IntegrationRegistry([FakeIntegration(available=False)])
        self.assertEqual(unavailable.get_native_tools(), [])

        self.assertEqual(
            registry.get_native_tools({CapabilityId("missing", "run")}),
            [],
        )

    def test_namespace_and_invalid_schema_are_rejected_at_startup(self):
        wrong_namespace = FakeIntegration(name="demo")
        wrong_namespace.name = "other"
        wrong_namespace.registered_tools = lambda: [RegisteredTool(
            spec=ToolSpec(
                capability=CapabilityId("demo", "run"),
                description="Wrong namespace.",
                input_schema={"type": "object", "properties": {}},
            ),
            handler=wrong_namespace._run,
        )]
        with self.assertRaises(ValueError):
            IntegrationRegistry([wrong_namespace])

        integration = FakeIntegration()
        integration.registered_tools = lambda: [RegisteredTool(
            spec=ToolSpec(
                capability=CapabilityId("demo", "run"),
                description="Invalid.",
                input_schema={"type": "not-a-real-type"},
            ),
            handler=integration._run,
        )]
        with self.assertRaises(ValueError):
            IntegrationRegistry([integration])

    def test_handler_exception_becomes_error_result(self):
        registry = IntegrationRegistry([FakeIntegration(raises=True)])
        result = registry.invoke(
            ToolCall(CapabilityId("demo", "run"), {"value": "x"}),
            InvocationContext("session-1", "hello"),
        )
        self.assertEqual(result.status.value, "error")

    def test_failing_context_provider_does_not_stop_other_providers(self):
        failing = FakeIntegration(name="broken")
        failing.context = lambda _invocation: (_ for _ in ()).throw(RuntimeError("boom"))
        registry = IntegrationRegistry([
            failing,
            FakeIntegration(name="healthy", context="state"),
        ])

        context = registry.collect_context(InvocationContext("session-1", "hello"), 20)

        self.assertEqual(context, "--- healthy ---\nstate")

    def test_context_contributions_are_bounded(self):
        registry = IntegrationRegistry([FakeIntegration(context="abcdef")])
        context = registry.collect_context(InvocationContext("session-1", "hello"), max_chars=3)

        self.assertEqual(context, "--- demo ---\nabc")


if __name__ == "__main__":
    unittest.main()
