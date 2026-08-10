import json
import unittest

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

    def emit(self, event, *args):
        self.emitted.append((event, args))

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
                "mindcraft__follow_player",
                "mindcraft__go_to_player",
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

        self.assertEqual(len(registry.get_native_tools()), 5)

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
        self.assertEqual(self.socket.emitted[-1][0], "execute-command")
        self.assertEqual(self.socket.emitted[-1][1][0][1]["request_id"], "op-123")

        self.socket.trigger("command-result", {
            "request_id": "op-123",
            "agent_name": "Andy",
            "command": '!goToPlayer("Biszeq", 2)',
            "status": "success",
            "message": "Arrived.",
            "interrupted": False,
            "timed_out": False,
            "state": {"gameplay": {"position": {"x": 1}}},
        })
        self.assertEqual(str(published[0].event), "mindcraft__command_completed")
        self.assertEqual(published[0].correlation_id, "op-123")


if __name__ == "__main__":
    unittest.main()
