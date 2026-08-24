import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.integrations import (
    CapabilityId,
    IntegrationRegistry,
    InvocationContext,
    MindcraftClient,
    MindcraftIntegration,
    ToolCall,
    RuntimeIntegration,
)


class FakeSocketClient:
    def __init__(self):
        self.connected = False
        self.handlers = {}
        self.emitted = []
        self.disconnect_calls = 0

    def on(self, event, handler):
        self.handlers[event] = handler

    def emit(self, event, *args, callback=None):
        self.emitted.append((event, args))
        if event == "integration-hello" and callback is not None:
            callback({
                "protocol_version": 1,
                "features": [
                    "typed_actions", "operation_lifecycle", "agent_events",
                    "vision_attachments",
                ],
            })

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    def trigger(self, event, *args):
        self.handlers[event](*args)


class MindcraftIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.socket = FakeSocketClient()
        self.client = MindcraftClient(
            url="http://localhost:8081/",
            socket_client=self.socket,
            recent_output_limit=2,
        )
        self.integration = MindcraftIntegration(self.client)
        self.registry = IntegrationRegistry([self.integration])

    def connect_with_agents(self, agents):
        self.socket.connected = True
        self.socket.trigger("connect")
        self.socket.trigger("agents-status", agents)

    def test_connect_subscribes_and_single_ready_agent_is_available(self):
        self.connect_with_agents([
            {
                "name": "Andy",
                "in_game": True,
                "socket_connected": True,
                "viewerPort": 3000,
            }
        ])

        self.assertEqual(self.client.url, "http://localhost:8081")
        self.assertIn(("listen-to-agents", ()), self.socket.emitted)
        self.assertEqual(
            {tool["function"]["name"] for tool in self.registry.get_native_tools()},
            {
                "mindcraft__collect_blocks",
                "mindcraft__collect_resource",
                "mindcraft__chop_tree",
                "mindcraft__capture_view",
                "mindcraft__follow_player",
                "mindcraft__go_to_player",
                "mindcraft__look_at_player",
                "mindcraft__look_at_position",
                "mindcraft__say",
                "mindcraft__send_message",
                "mindcraft__stop",
            },
        )

    def test_send_message_uses_mindcraft_wire_payload(self):
        self.connect_with_agents([
            {"name": "Andy", "in_game": True, "socket_connected": True}
        ])

        result = self.registry.invoke(
            ToolCall(
                CapabilityId("mindcraft", "send_message"),
                {"message": "Collect ten oak logs."},
            ),
            InvocationContext("session-1", "Ask Andy to collect wood"),
        )

        self.assertEqual(result.status.value, "success")
        self.assertIn("Andy", result.content)
        self.assertEqual(
            self.socket.emitted[-1],
            (
                "send-message",
                (("Andy", {"from": "local_assistant", "message": "Collect ten oak logs."}),),
            ),
        )

    def test_direct_actions_emit_forced_commands_without_natural_language(self):
        self.connect_with_agents([
            {"name": "Andy", "in_game": True, "socket_connected": True}
        ])
        cases = [
            ("stop", {}, "!endGoal"),
            (
                "go_to_player",
                {"player_name": "Biszeq", "closeness": 2},
                '!goToPlayer("Biszeq", 2)',
            ),
            (
                "follow_player",
                {"player_name": "Biszeq", "follow_distance": 3.5},
                '!followPlayer("Biszeq", 3.5)',
            ),
            (
                "collect_blocks",
                {"block_type": "oak_log", "count": 10},
                '!collectBlocks("oak_log", 10)',
            ),
        ]

        for action, arguments, command in cases:
            with self.subTest(action=action):
                result = self.registry.invoke(
                    ToolCall(CapabilityId("mindcraft", action), arguments),
                    InvocationContext("session-1", "control Andy"),
                )
                self.assertEqual(result.status.value, "success")
                self.assertEqual(
                    self.socket.emitted[-1],
                    (
                        "send-message",
                        (("Andy", {"from": "local_assistant", "message": command}),),
                    ),
                )

    def test_direct_action_schemas_reject_command_injection_and_bad_bounds(self):
        self.connect_with_agents([
            {"name": "Andy", "in_game": True, "socket_connected": True}
        ])
        emitted_before = list(self.socket.emitted)
        invalid_calls = [
            ToolCall(
                CapabilityId("mindcraft", "go_to_player"),
                {"player_name": 'Biszeq")!stop', "closeness": 2},
            ),
            ToolCall(
                CapabilityId("mindcraft", "follow_player"),
                {"player_name": "Biszeq", "follow_distance": -1},
            ),
            ToolCall(
                CapabilityId("mindcraft", "collect_blocks"),
                {"block_type": 'oak_log")!stop', "count": 1},
            ),
            ToolCall(
                CapabilityId("mindcraft", "collect_blocks"),
                {"block_type": "oak_log", "count": 0},
            ),
        ]

        for call in invalid_calls:
            with self.subTest(capability=str(call.capability), arguments=call.arguments):
                result = self.registry.invoke(
                    call,
                    InvocationContext("session-1", "control Andy"),
                )
                self.assertEqual(result.status.value, "error")

        self.assertEqual(self.socket.emitted, emitted_before)

    def test_disconnected_offline_and_ambiguous_agents_hide_capability(self):
        self.assertEqual(self.registry.get_native_tools(), [])

        self.connect_with_agents([
            {"name": "Andy", "in_game": False, "socket_connected": True}
        ])
        self.assertEqual(self.registry.get_native_tools(), [])

        self.socket.trigger("agents-status", [
            {"name": "Andy", "in_game": True, "socket_connected": True},
            {"name": "Sam", "in_game": True, "socket_connected": True},
        ])
        self.assertEqual(self.registry.get_native_tools(), [])

    def test_explicit_agent_resolves_multi_agent_server(self):
        client = MindcraftClient(
            url="http://localhost:8081",
            agent_name="Sam",
            socket_client=self.socket,
        )
        registry = IntegrationRegistry([MindcraftIntegration(client)])
        self.socket.connected = True
        self.socket.trigger("connect")
        self.socket.trigger("agents-status", [
            {"name": "Andy", "in_game": True, "socket_connected": True},
            {"name": "Sam", "in_game": True, "socket_connected": True},
        ])

        self.assertEqual(len(registry.get_native_tools()), 11)

    def test_context_contains_world_state_and_recent_output(self):
        self.connect_with_agents([
            {"name": "Andy", "in_game": True, "socket_connected": True}
        ])
        self.socket.trigger("state-update", {
            "Andy": {
                "gameplay": {"position": {"x": 1, "y": 64, "z": 2}, "health": 20},
                "action": {"current": "Collecting oak logs", "kind": "acting"},
                "inventory": {"counts": {"oak_log": 4}},
                "modes": {"summary": "intentionally omitted from context"},
            }
        })
        self.socket.trigger("bot-output", "Andy", "first")
        self.socket.trigger("bot-output", "Andy", "second")
        self.socket.trigger("bot-output", "Andy", "third")

        contribution = self.integration.context(InvocationContext("s", "status"))
        payload = json.loads(contribution.content.split("\n", 1)[1])

        self.assertEqual(payload["target_agent"], "Andy")
        self.assertEqual(payload["world"]["gameplay"]["health"], 20)
        self.assertNotIn("modes", payload["world"])
        self.assertEqual(payload["recent_output"], ["second", "third"])

    def test_malformed_updates_are_ignored(self):
        self.connect_with_agents("not-a-list")
        self.socket.trigger("state-update", "not-an-object")

        snapshot = self.client.context_snapshot()

        self.assertEqual(snapshot["agents"], [])
        self.assertNotIn("world", snapshot)

    def test_context_can_be_disabled(self):
        integration = MindcraftIntegration(self.client, context_enabled=False)

        self.assertIsNone(integration.context(InvocationContext("s", "hello")))

    def test_registry_close_disconnects_client(self):
        self.socket.connected = True

        self.registry.close()

        self.assertEqual(self.socket.disconnect_calls, 1)

    def test_connection_failures_warn_once_and_back_off_until_recovery(self):
        client = MindcraftClient(
            url="http://localhost:8081",
            reconnect_delay_s=2,
            reconnect_max_delay_s=5,
            socket_client=self.socket,
        )

        with self.assertLogs("mindcraft_integration", level="WARNING") as first_logs:
            first_delay = client._record_connection_failure(ConnectionError("offline"))
        with self.assertLogs("mindcraft_integration", level="DEBUG") as repeated_logs:
            second_delay = client._record_connection_failure(ConnectionError("offline"))
        third_delay = client._record_connection_failure(ConnectionError("offline"))

        self.assertEqual(first_delay, 2)
        self.assertEqual(second_delay, 4)
        self.assertEqual(third_delay, 5)
        self.assertEqual(sum("WARNING" in line for line in first_logs.output), 1)
        self.assertFalse(any("WARNING" in line for line in repeated_logs.output))

        self.socket.trigger("connect")
        with self.assertLogs("mindcraft_integration", level="WARNING") as recovered_logs:
            reset_delay = client._record_connection_failure(ConnectionError("offline again"))

        self.assertEqual(reset_delay, 2)
        self.assertEqual(sum("WARNING" in line for line in recovered_logs.output), 1)

    def test_event_enabled_command_is_correlated_and_publishes_completion(self):
        integration = MindcraftIntegration(self.client, events_enabled=True)
        registry = IntegrationRegistry([RuntimeIntegration(), integration])
        published = []
        integration._publisher = lambda event: published.append(event) or event.event_id
        self.client.set_command_result_handler(integration._publish_command_result)
        self.client.set_operation_event_handler(integration._publish_operation_event)
        self.connect_with_agents([
            {"name": "Andy", "in_game": True, "socket_connected": True}
        ])

        result = registry.invoke(
            ToolCall(CapabilityId("mindcraft", "go_to_player"), {
                "player_name": "Biszeq", "closeness": 2,
            }),
            InvocationContext("session-1", "go", invocation_id="op-123"),
        )

        self.assertEqual(result.status.value, "pending")
        self.assertEqual(result.operation_id, "op-123")
        self.assertEqual(self.socket.emitted[-1][0], "execute-action")
        self.assertEqual(self.socket.emitted[-1][1][0][1]["operation_id"], "op-123")

        self.socket.trigger("operation-event", {
            "operation_id": "op-123",
            "agent_name": "Andy",
            "action": "go_to_player",
            "status": "completed",
            "message": "Arrived.",
            "interrupted": False,
            "timed_out": False,
            "terminal": True,
            "state": {"gameplay": {"position": {"x": 1}}},
        })
        self.assertEqual(str(published[0].event), "mindcraft__command_completed")
        self.assertEqual(published[0].correlation_id, "op-123")

    def test_exact_agent_events_bind_only_after_session_controls_bot(self):
        integration = MindcraftIntegration(
            self.client,
            events_enabled=True,
            autonomous_events=("critical_health",),
        )
        published = []
        integration._publisher = lambda event: published.append(event) or event.event_id
        self.client.set_agent_event_handler(integration._publish_agent_event)
        self.connect_with_agents([
            {"name": "Andy", "in_game": True, "socket_connected": True}
        ])

        common = {
            "protocol_version": 1,
            "sequence": 1,
            "agent_name": "Andy",
            "occurred_at": "2026-08-11T10:00:00.000Z",
            "state": {"gameplay": {"health": 4}},
        }
        self.socket.trigger("agent-event", {
            **common,
            "event_id": "boot:1",
            "event": "critical_health",
            "payload": {"health": 4},
        })
        self.assertIsNone(published[-1].session_id)

        integration._send_action(
            "go_to_player",
            {"player_name": "Biszeq", "closeness": 2},
            "Go",
            InvocationContext("session-1", "go", invocation_id="op-1"),
        )
        self.socket.trigger("agent-event", {
            **common,
            "sequence": 2,
            "event_id": "boot:2",
            "event": "critical_health",
            "payload": {"health": 3},
        })
        self.socket.trigger("agent-event", {
            **common,
            "sequence": 3,
            "event_id": "boot:3",
            "event": "player_spoke",
            "payload": {"player_name": "Alex", "message": "hello"},
        })

        self.assertEqual(published[-2].session_id, "session-1")
        self.assertIsNone(published[-1].session_id)
        player_spec = next(
            spec for spec in integration.registered_events()
            if str(spec.event) == "mindcraft__player_spoke"
        )
        self.assertEqual(
            {str(capability) for capability in player_spec.allowed_capabilities},
            {"mindcraft__say"},
        )

    def test_external_controller_mode_hides_planner_delegation(self):
        self.connect_with_agents([
            {"name": "Andy", "in_game": True, "socket_connected": True}
        ])
        self.client._on_protocol_manifest({
            "protocol_version": 1,
            "features": ["typed_actions"],
            "agent_controller_modes": {"Andy": "external"},
        })

        names = {tool["function"]["name"] for tool in self.registry.get_native_tools()}

        self.assertNotIn("mindcraft__send_message", names)
        self.assertIn("mindcraft__collect_resource", names)

    def test_vision_operation_persists_verified_attachment(self):
        image = b"jpeg-payload"
        digest = hashlib.sha256(image).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            integration = MindcraftIntegration(
                self.client,
                events_enabled=True,
                attachment_dir=directory,
            )
            published = []
            integration._publisher = lambda event: published.append(event) or event.event_id
            self.client.set_operation_event_handler(integration._publish_operation_event)
            self.socket.trigger("operation-event", {
                "operation_id": "op-image",
                "agent_name": "Andy",
                "action": "capture_view",
                "status": "completed",
                "message": "Captured.",
                "terminal": True,
                "data": {
                    "attachment": {
                        "name": "view.jpg",
                        "mime_type": "image/jpeg",
                        "sha256": digest,
                        "data_base64": base64.b64encode(image).decode("ascii"),
                    }
                },
                "state": {},
            })

            attachment = published[0].attachments[0]
            self.assertEqual(attachment.sha256, digest)
            self.assertEqual(Path(attachment.storage_path).read_bytes(), image)

    def test_reconnect_recovers_terminal_pending_operation(self):
        class FakeOperationStore:
            def __init__(self):
                self.finished = []

            def pending_operations(self, prefix):
                self.prefix = prefix
                return [SimpleNamespace(
                    invocation_id="op-recovered",
                    capability="mindcraft__collect_resource",
                )]

            def has_event_deduplication_key(self, _event_type, _key):
                return False

            def finish_operation(self, operation_id, status, result):
                self.finished.append((operation_id, status, result))

        store = FakeOperationStore()
        integration = MindcraftIntegration(
            self.client,
            events_enabled=True,
            operation_store=store,
        )
        published = []
        integration._publisher = lambda event: published.append(event) or event.event_id
        self.client.query_operation = lambda _operation_id: {
            "operation_id": "op-recovered",
            "agent_name": "Andy",
            "action": "collect_resource",
            "status": "completed",
            "message": "Collected ten coal ore.",
            "terminal": True,
            "state": {},
        }

        integration._reconcile_pending_operations()

        self.assertEqual(store.prefix, "mindcraft__")
        self.assertEqual(str(published[0].event), "mindcraft__command_completed")
        self.assertEqual(store.finished[0][1], "success")


if __name__ == "__main__":
    unittest.main()
