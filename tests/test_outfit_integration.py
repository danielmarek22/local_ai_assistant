import unittest

from app.integrations import (
    AvatarWardrobe,
    CapabilityId,
    IntegrationRegistry,
    InvocationContext,
    OutfitIntegration,
    ToolCall,
)


class OutfitIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.wardrobe = AvatarWardrobe(
            {
                "casual": "/static/avatars/casual.vrm",
                "pajamas": "/static/avatars/pajamas.vrm",
            },
            "casual",
        )
        self.integration = OutfitIntegration(self.wardrobe)
        self.registry = IntegrationRegistry([self.integration])

    def test_native_schema_uses_discovered_outfits(self):
        schema = self.registry.get_native_tools()[0]["function"]["parameters"]
        self.assertEqual(schema["properties"]["outfit"]["enum"], ["casual", "pajamas"])

    def test_change_updates_runtime_state_and_emits_typed_effect(self):
        result = self.registry.invoke(
            ToolCall(CapabilityId("outfit", "change"), {"outfit": "pajamas"}),
            InvocationContext("session-1", "wear pajamas"),
        )

        self.assertEqual(result.status.value, "success")
        self.assertEqual(self.wardrobe.current_outfit, "pajamas")
        self.assertEqual(result.effects[0].outfit, "pajamas")
        self.assertEqual(result.effects[0].url, "/static/avatars/pajamas.vrm")

    def test_current_outfit_is_runtime_context_not_a_belief(self):
        contribution = self.integration.context(InvocationContext("session-1", "hello"))
        self.assertEqual(contribution.source, "outfit")
        self.assertEqual(contribution.content, "Astra's current outfit is casual.")

    def test_selecting_current_outfit_is_idempotent(self):
        result = self.registry.invoke(
            ToolCall(CapabilityId("outfit", "change"), {"outfit": "casual"}),
            InvocationContext("session-1", "stay casual"),
        )
        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.effects, ())

    def test_unknown_outfit_is_rejected_by_schema(self):
        result = self.registry.invoke(
            ToolCall(CapabilityId("outfit", "change"), {"outfit": "https://bad.example/x.vrm"}),
            InvocationContext("session-1", "change"),
        )
        self.assertEqual(result.status.value, "error")
        self.assertEqual(result.diagnostics["repository_accessed"], False)

    def test_empty_wardrobe_exposes_no_tool_or_context(self):
        integration = OutfitIntegration(AvatarWardrobe({}, ""))
        registry = IntegrationRegistry([integration])
        self.assertEqual(registry.get_native_tools(), [])
        self.assertIsNone(integration.context(InvocationContext("session-1", "hello")))


if __name__ == "__main__":
    unittest.main()
