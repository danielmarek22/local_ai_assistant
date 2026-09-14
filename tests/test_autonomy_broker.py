import asyncio
import tempfile
import unittest

from app.autonomy import AutonomyRuntime, AutonomyStore, EventTurnOutcome, IntegrationEventBroker
from app.integrations import EventId, EventSpec, IntegrationEvent, IntegrationRegistry


class EventIntegration:
    name = "demo"

    def registered_tools(self):
        return []

    def registered_events(self):
        return [EventSpec(
            event=EventId("demo", "finished"),
            description="Finished.",
            payload_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )]


class AutonomyRuntimeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_releases_broker_and_store_when_registry_close_fails(self):
        close_order = []
        runtime = AutonomyRuntime.__new__(AutonomyRuntime)

        class FailingRegistry:
            def close(self):
                close_order.append("registry")
                raise RuntimeError("registry failed")

        class Broker:
            async def close(self):
                close_order.append("broker")

        class Store:
            def close(self):
                close_order.append("store")

        runtime.registry = FailingRegistry()
        runtime.broker = Broker()
        runtime.store = Store()

        with self.assertRaisesRegex(RuntimeError, "registry failed"):
            await runtime.close()

        self.assertEqual(close_order, ["registry", "broker", "store"])


class AutonomyBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = AutonomyStore(f"{self.temp.name}/assistant.db")
        self.registry = IntegrationRegistry([EventIntegration()])
        self.handled = []

        async def handle(record, _spec):
            self.handled.append(record)
            return EventTurnOutcome("done")

        self.broker = IntegrationEventBroker(self.registry, self.store, handle)
        await self.broker.start()

    async def asyncTearDown(self):
        await self.broker.close()
        self.store.close()
        self.temp.cleanup()

    async def test_valid_event_is_processed_once(self):
        event = IntegrationEvent(EventId("demo", "finished"), {"value": 1}, "session-1")
        await self.broker.publish_async(event)
        await asyncio.wait_for(self.broker._queue.join(), timeout=1)

        self.assertEqual(len(self.handled), 1)
        self.assertEqual(self.store.get_event(event.event_id).status, "completed")

    async def test_invalid_payload_never_enters_journal(self):
        event = IntegrationEvent(EventId("demo", "finished"), {"value": "bad"}, "session-1")
        with self.assertRaises(ValueError):
            await self.broker.publish_async(event)
        self.assertIsNone(self.store.get_event(event.event_id))

    async def test_correlation_inherits_operation_session(self):
        self.store.begin_operation("op-1", "demo__run", "session-42", None, None, None)
        event = IntegrationEvent(
            EventId("demo", "finished"),
            {"value": 1},
            correlation_id="op-1",
        )
        await self.broker.publish_async(event)
        await asyncio.wait_for(self.broker._queue.join(), timeout=1)

        self.assertEqual(self.handled[0].event.session_id, "session-42")

    async def test_paused_events_remain_pending_until_resume(self):
        await self.broker.set_paused(True)
        event = IntegrationEvent(EventId("demo", "finished"), {"value": 1}, "session-1")
        await self.broker.publish_async(event)
        await asyncio.sleep(0)
        self.assertEqual(self.store.get_event(event.event_id).status, "pending")

        await self.broker.set_paused(False)
        await asyncio.wait_for(self.broker._queue.join(), timeout=1)
        self.assertEqual(self.store.get_event(event.event_id).status, "completed")

    async def test_chain_limit_discards_descendant_without_model_turn(self):
        self.broker.max_chain_events = 1
        root = IntegrationEvent(EventId("demo", "finished"), {"value": 1}, "session-1")
        await self.broker.publish_async(root)
        await asyncio.wait_for(self.broker._queue.join(), timeout=1)
        child = IntegrationEvent(
            EventId("demo", "finished"),
            {"value": 2},
            "session-1",
            causation_id=root.event_id,
            root_event_id=root.event_id,
        )

        await self.broker.publish_async(child)
        await asyncio.wait_for(self.broker._queue.join(), timeout=1)

        self.assertEqual(len(self.handled), 1)
        self.assertEqual(self.store.get_event(child.event_id).status, "discarded")
        self.assertEqual(
            self.store.get_event(child.event_id).notification["delivery"],
            "text",
        )


if __name__ == "__main__":
    unittest.main()
