import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from datetime import datetime, timezone

from app.autonomy import AutonomyStore
from app.core.tool_executor import ToolExecutor
from app.integrations import (
    CapabilityId, IntegrationRegistry, RegisteredTool, ToolCall, ToolResult, ToolSpec,
    EventId,
    EventSpec,
    IntegrationEvent,
    ReplayPolicy,
)


class AutonomyStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = AutonomyStore(f"{self.temp.name}/assistant.db")
        self.spec = EventSpec(
            event=EventId("demo", "finished"),
            description="Finished.",
            payload_schema={"type": "object", "properties": {}},
            coalesce_window_s=10,
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_event_round_trip_completion_and_recent_context(self):
        event = IntegrationEvent(
            event=self.spec.event,
            payload={"value": 1},
            session_id="session-1",
        )
        self.store.append_event(event, self.spec)
        claimed = self.store.claim_event(event.event_id)
        self.assertEqual(claimed.status, "processing")

        self.store.complete_event(event.event_id, "It worked.", {"delivery": "text"})

        completed = self.store.get_event(event.event_id)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.notification, {"delivery": "text"})
        self.assertIn("It worked", self.store.recent_context("session-1", 100))

    def test_pending_events_coalesce_by_type_session_and_key(self):
        first = IntegrationEvent(
            event=self.spec.event,
            payload={"value": 1},
            session_id="session-1",
            deduplication_key="same",
        )
        second = IntegrationEvent(
            event=self.spec.event,
            payload={"value": 2},
            session_id="session-1",
            deduplication_key="same",
        )
        first_id = self.store.append_event(first, self.spec)
        second_id = self.store.append_event(second, self.spec)

        self.assertEqual(first_id, second_id)
        self.assertEqual(self.store.get_event(first_id).event.payload["value"], 2)

    def test_restart_replays_only_explicitly_safe_events(self):
        safe_spec = EventSpec(
            event=self.spec.event,
            description="Safe.",
            payload_schema=self.spec.payload_schema,
            replay_policy=ReplayPolicy.SAFE,
        )
        safe = IntegrationEvent(self.spec.event, {}, "session-1")
        unsafe = IntegrationEvent(self.spec.event, {}, "session-1")
        self.store.append_event(safe, safe_spec)
        self.store.append_event(unsafe, self.spec)
        self.store.claim_event(safe.event_id)
        self.store.claim_event(unsafe.event_id)

        replayed, failed = self.store.recover_interrupted()

        self.assertEqual((replayed, failed), (1, 1))
        self.assertEqual(self.store.get_event(safe.event_id).status, "pending")
        self.assertEqual(self.store.get_event(unsafe.event_id).status, "failed")

    def test_operation_and_pause_state_are_durable(self):
        self.store.begin_operation("op-1", "demo__run", "session-1", None, None, None)
        self.store.finish_operation("op-1", "pending", "accepted")
        self.store.set_paused(True)

        self.assertEqual(self.store.get_operation("op-1").session_id, "session-1")
        self.assertEqual(self.store.get_operation("op-1").status, "pending")
        self.assertTrue(self.store.is_paused())

    def test_late_pending_update_does_not_overwrite_terminal_operation(self):
        self.store.begin_operation("op-1", "mindcraft__say", "session-1", None, None, None)
        self.store.finish_operation("op-1", "success", "done")
        self.store.finish_operation("op-1", "pending", "accepted")

        self.assertEqual(self.store.get_operation("op-1").status, "success")
        self.assertEqual(self.store.pending_operations("mindcraft__"), [])

    def operation_row(self):
        return dict(self.store._conn.execute(
            "SELECT * FROM integration_operations WHERE invocation_id = 'op-1'",
        ).fetchone())

    def begin(self):
        self.store.begin_operation("op-1", "demo__run", "session-1", None, None, None)

    def test_duplicate_completion_is_an_exact_noop(self):
        self.begin()
        self.store.finish_operation("op-1", "success", "done")
        before = self.operation_row()
        self.store.finish_operation("op-1", "success", "done")
        self.assertEqual(self.operation_row(), before)
        self.assertEqual(self.store.operation_conflicts("op-1"), [])

    def test_all_terminal_outcomes_reject_late_results(self):
        terminals = ("success", "error", "denied", "unavailable", "cancelled")
        for terminal in terminals:
            with self.subTest(terminal=terminal):
                operation_id = f"op-{terminal}"
                self.store.begin_operation(operation_id, "demo__run", "session-1", None, None, None)
                self.store.finish_operation(operation_id, "pending", "accepted")
                self.store.finish_operation(operation_id, terminal, "original")
                for late_status in (*terminals, "pending"):
                    self.store.finish_operation(operation_id, late_status, "late")
                self.assertEqual(self.store.get_operation(operation_id).status, terminal)
                conflicts = self.store.operation_conflicts(operation_id)
                self.assertEqual(len(conflicts), 6)
                self.assertTrue(all(item["accepted_result"] == "original" for item in conflicts))
                self.assertEqual([item["attempted_status"] for item in conflicts], [*terminals, "pending"])

    def test_repeated_begin_preserves_existing_record_at_every_stage(self):
        self.begin()
        for status in ("running", "pending", "success"):
            if status != "running":
                self.store.finish_operation("op-1", status, "accepted")
            before = self.operation_row()
            self.begin()
            self.assertEqual(self.operation_row(), before)

    def test_reused_id_with_different_identity_is_rejected(self):
        self.begin()
        before = self.operation_row()
        identity = ["demo__run", "session-1", None, None, None]
        for index in range(len(identity)):
            conflicting = identity.copy()
            conflicting[index] = "different"
            with self.assertRaisesRegex(ValueError, "different invocation"):
                self.store.begin_operation("op-1", *conflicting)
            self.assertEqual(self.operation_row(), before)

    def test_invalid_status_cannot_change_operation(self):
        self.begin()
        before = self.operation_row()
        for status in ("running", "completed", "typo"):
            with self.assertRaisesRegex(ValueError, "Invalid operation status"):
                self.store.finish_operation("op-1", status)
        self.assertEqual(self.operation_row(), before)

    def test_conflicts_survive_reopening_and_concurrent_completions(self):
        self.begin()
        other = AutonomyStore(self.store.path)
        self.addCleanup(other.close)
        barrier = Barrier(2)

        def complete(store, status):
            barrier.wait(timeout=5)
            store.finish_operation("op-1", status, status)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(complete, store, status) for store, status in
                       ((self.store, "success"), (other, "error"))]
            for future in futures:
                future.result(timeout=10)
        reopened = AutonomyStore(self.store.path)
        self.addCleanup(reopened.close)
        conflicts = reopened.operation_conflicts("op-1")
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["accepted_status"], reopened.get_operation("op-1").status)
        self.assertNotEqual(conflicts[0]["attempted_status"], conflicts[0]["accepted_status"])

    def test_executor_preserves_external_completion_before_handler_returns(self):
        observed_ids = []

        def handler(arguments, context):
            observed_ids.append(context.invocation_id)
            self.store.finish_operation(context.invocation_id, "success", "external completion")
            return ToolResult.pending("accepted", context.invocation_id)

        capability = CapabilityId("demo", "run")
        tool = RegisteredTool(ToolSpec(capability, "Demo", {"type": "object"}), handler)
        registry = IntegrationRegistry([SimpleNamespace(name="demo", registered_tools=lambda: [tool])])
        executor = ToolExecutor(registry, operation_store=self.store)
        list(executor.execute(ToolCall(capability, {}), "session-1", "run"))
        self.assertEqual(len(observed_ids), 1)
        self.assertEqual(self.store.get_operation(observed_ids[0]).status, "success")
        conflict = self.store.operation_conflicts(observed_ids[0])[0]
        self.assertEqual(conflict["accepted_result"], "external completion")
        self.assertEqual(conflict["attempted_status"], "pending")


if __name__ == "__main__":
    unittest.main()
