import asyncio
import json
import unittest

from app.autonomy import SessionTurnCoordinator
from app.services.connection_hub import SessionConnectionHub


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, value):
        self.sent.append(json.loads(value))


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_waiter_runs_before_next_event(self):
        coordinator = SessionTurnCoordinator()
        order = []
        release = asyncio.Event()

        async def first_event():
            async with coordinator.event_turn("a"):
                order.append("event-1")
                await release.wait()

        async def user():
            async with coordinator.user_turn("b"):
                order.append("user")

        async def second_event():
            async with coordinator.event_turn("c"):
                order.append("event-2")

        first = asyncio.create_task(first_event())
        await asyncio.sleep(0)
        user_task = asyncio.create_task(user())
        await asyncio.sleep(0)
        second = asyncio.create_task(second_event())
        release.set()
        await asyncio.gather(first, user_task, second)

        self.assertEqual(order, ["event-1", "user", "event-2"])

    async def test_global_model_limit_prevents_cross_session_overlap(self):
        coordinator = SessionTurnCoordinator(global_concurrency=1)
        active = 0
        maximum = 0

        async def run(session):
            nonlocal active, maximum
            async with coordinator.user_turn(session):
                active += 1
                maximum = max(maximum, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(run("a"), run("b"))
        self.assertEqual(maximum, 1)


class ConnectionHubTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_metadata_is_added_to_assistant_payloads(self):
        hub = SessionConnectionHub()
        ws = FakeWebSocket()
        hub.register("session-1", "connection-1", ws)
        hub.set_turn("connection-1", "turn-1", "user")

        await hub.send_websocket(ws, {"type": "assistant_chunk", "content": "hello"})

        self.assertEqual(ws.sent[0]["turn_id"], "turn-1")
        self.assertEqual(ws.sent[0]["origin"], "user")

    async def test_approval_is_bound_to_selected_connection(self):
        hub = SessionConnectionHub()
        ws = FakeWebSocket()
        hub.register("session-1", "connection-1", ws)
        task = asyncio.create_task(hub.request_approval(
            "session-1", {"tool": "shell__execute"}, timeout_seconds=1,
        ))
        await asyncio.sleep(0)
        approval_id = ws.sent[0]["approval_id"]

        self.assertFalse(hub.resolve_approval("wrong", {
            "approval_id": approval_id, "approved": True,
        }))
        self.assertTrue(hub.resolve_approval("connection-1", {
            "approval_id": approval_id, "approved": True,
        }))
        self.assertTrue(await task)

    async def test_headless_approval_is_denied(self):
        hub = SessionConnectionHub()
        self.assertFalse(await hub.request_approval(
            "missing", {"tool": "shell__execute"}, timeout_seconds=0.01,
        ))

    async def test_non_boolean_approval_decisions_fail_closed(self):
        for value in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(value=value):
                hub = SessionConnectionHub()
                ws = FakeWebSocket()
                hub.register("session-1", "connection-1", ws)
                task = asyncio.create_task(hub.request_approval(
                    "session-1", {"tool": "shell__execute"}, timeout_seconds=1,
                ))
                await asyncio.sleep(0)
                approval_id = ws.sent[0]["approval_id"]

                self.assertTrue(hub.resolve_approval("connection-1", {
                    "approval_id": approval_id,
                    "approved": value,
                }))
                self.assertFalse(await task)


if __name__ == "__main__":
    unittest.main()
