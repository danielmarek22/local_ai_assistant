import tempfile
import unittest
from datetime import datetime, timezone

from app.autonomy import AutonomyStore
from app.integrations import (
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


if __name__ == "__main__":
    unittest.main()
