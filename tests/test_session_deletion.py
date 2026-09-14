import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.autonomy.coordinator import SessionTurnCoordinator
from app.autonomy.store import AutonomyStore
from app.memory.chat_history import ChatHistoryStore
from app.memory.summary_store import SummaryStore
from app.storage.database import Database
from app.transport.connection_hub import SessionConnectionHub
from app import server
from app.integrations import EventId, EventSpec, IntegrationEvent, ReplayPolicy
from app.beliefs.repository import BeliefRepository
from app.beliefs.models import BeliefMutation, CandidateOperation, EpistemicStatus, SubjectKind, VisibilityPolicy


class SessionDeletionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / "assistant.db")
        self.db = Database(self.path)
        self.addCleanup(self.db.close)
        self.history = ChatHistoryStore(
            self.db, SimpleNamespace(episodic_collection=Mock()),
            uploads_root=str(Path(self.temp.name) / "uploads"),
        )
        self.summary = SummaryStore(self.db)
        self.autonomy = AutonomyStore(self.path)
        self.addCleanup(self.autonomy.close)
        self.coordinator = SessionTurnCoordinator()
        self.hub = SessionConnectionHub()
        self.app = SimpleNamespace(state=SimpleNamespace(
            orchestrator=SimpleNamespace(
                history=self.history, summary_store=self.summary,
                autonomy_runtime=SimpleNamespace(coordinator=self.coordinator, store=self.autonomy),
            ), connection_hub=self.hub,
        ))
        self.request = SimpleNamespace(app=self.app)
        self.history.add("session", "user", "Question")

    async def test_deletion_waits_for_active_turn_then_removes_its_answer(self):
        async with self.coordinator.user_turn("session"):
            deletion = asyncio.create_task(server.delete_session("session", self.request))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(deletion.done())
            self.history.add("session", "assistant", "Answer")
        result = await asyncio.wait_for(deletion, 1)
        self.assertTrue(result["deleted"])
        self.assertFalse(self.history.session_exists("session"))

    async def test_deleted_session_rejects_late_history_and_summary_writes_after_restart(self):
        await server.delete_session("session", self.request)
        reopened = Database(self.path)
        self.addCleanup(reopened.close)
        history = ChatHistoryStore(
            reopened, SimpleNamespace(episodic_collection=Mock()),
            uploads_root=str(Path(self.temp.name) / "uploads"),
        )
        with self.assertRaisesRegex(ValueError, "deleted"):
            history.add("session", "assistant", "Late answer")
        with self.assertRaisesRegex(ValueError, "deleted"):
            SummaryStore(reopened).set("session", "Late summary", 2)
        self.assertFalse(history.session_exists("session"))

    async def test_queued_turns_are_rejected_and_other_sessions_can_continue(self):
        async def turn(context):
            async with context:
                self.fail("Deleted session admitted queued work")

        async with self.coordinator.user_turn("session"):
            user = asyncio.create_task(turn(self.coordinator.user_turn("session")))
            event = asyncio.create_task(turn(self.coordinator.event_turn("session")))
            await asyncio.sleep(0)
            deletion = asyncio.create_task(server.delete_session("session", self.request))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            for queued in (user, event):
                with self.assertRaisesRegex(ValueError, "deleted"):
                    await asyncio.wait_for(queued, 1)
        await asyncio.wait_for(deletion, 1)
        async with self.coordinator.event_turn("other"):
            self.history.add("other", "assistant", "Unaffected")
        self.assertTrue(self.history.session_exists("other"))

    async def test_cancelled_caller_does_not_cancel_shared_deletion(self):
        async with self.coordinator.user_turn("session"):
            first = asyncio.create_task(server.delete_session("session", self.request))
            second = asyncio.create_task(server.delete_session("session", self.request))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(len(self.app.state.session_deletion_tasks), 1)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertFalse(second.done())
        self.assertTrue((await asyncio.wait_for(second, 1))["deleted"])
        self.assertFalse(self.history.session_exists("session"))
        self.assertTrue((await server.delete_session("session", self.request))["deleted"])

    async def test_queued_events_and_operations_stay_retired_after_restart(self):
        spec = EventSpec(EventId("demo", "finished"), "Finished", {}, replay_policy=ReplayPolicy.SAFE)
        pending = IntegrationEvent(spec.event, {}, "session")
        processing = IntegrationEvent(spec.event, {}, "session")
        for event in (pending, processing):
            self.autonomy.append_event(event, spec)
        self.autonomy.claim_event(processing.event_id)
        self.autonomy.begin_operation("op", "demo__run", "session", None, None, None)
        await server.delete_session("session", self.request)
        reopened = AutonomyStore(self.path)
        self.addCleanup(reopened.close)
        reopened.recover_interrupted()
        self.assertEqual(reopened.pending_event_ids(), [])
        for event in (pending, processing):
            self.assertIsNone(reopened.claim_event(event.event_id))
            reopened.complete_event(event.event_id, "Late completion", {})
            self.assertEqual(reopened.get_event(event.event_id).status, "discarded")
        reopened.finish_operation("op", "completed", "Late callback")
        self.assertEqual(reopened.get_operation("op").status, "cancelled")
        with self.assertRaisesRegex(ValueError, "deleted"):
            reopened.append_event(IntegrationEvent(spec.event, {}, "session"), spec)
        with self.assertRaisesRegex(ValueError, "deleted"):
            reopened.begin_operation("late", "demo__run", "session", None, None, None)

    async def test_connections_close_and_buffered_replay_is_removed(self):
        socket = SimpleNamespace(close=AsyncMock(), send_json=AsyncMock())
        other = SimpleNamespace(close=AsyncMock(), send_json=AsyncMock())
        await self.hub.broadcast("session", {"type": "assistant_end"})
        self.hub.register("session", "connection", socket)
        self.hub.register("other", "other-connection", other)
        approval = asyncio.get_running_loop().create_future()
        self.hub._approvals["approval"] = ("connection", approval)
        await server.delete_session("session", self.request)
        socket.close.assert_awaited_once_with(code=4004, reason="Conversation deleted")
        other.close.assert_not_awaited()
        self.assertFalse(await approval)
        self.assertFalse(self.hub.has_session("session"))
        self.assertNotIn("session", self.hub._pending_assistant_frames)
        await self.hub.broadcast("session", {"type": "assistant_end"})
        self.assertNotIn("session", self.hub._pending_assistant_frames)
        with self.assertRaisesRegex(ValueError, "deleted"):
            self.hub.register("session", "late", socket)

    async def test_canonical_cleanup_preserves_global_beliefs_and_rejects_late_updates(self):
        repository = BeliefRepository(self.db)
        self.addCleanup(repository.close)
        now = datetime.now(timezone.utc)
        mutations = [BeliefMutation(
            operation=CandidateOperation.ASSERT, belief_id=None, visibility=visibility,
            source_session_id="session", subject_id="person", subject_kind=SubjectKind.PERSON,
            subject_display_name="Person", predicate="beverage", value="tea",
            epistemic_status=EpistemicStatus.SELF_REPORT, source_sender_id="person",
            source_sender_display_name="Person", source_sender_type="human",
            source_input_source="direct",
        ) for visibility in (VisibilityPolicy.SESSION_CURRENT, VisibilityPolicy.AGENT_CURRENT)]
        repository.apply_mutations(owner_agent_id="agent", source_message_id=1,
                                   extractor_version="test", mutations=mutations, now=now)
        self.summary.set("session", "Saved summary", 1)
        await server.delete_session("session", self.request)
        with self.db.connection() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM conversation_summary").fetchone()[0], 0)
            rows = conn.execute("SELECT visibility FROM beliefs").fetchall()
        self.assertEqual([row[0] for row in rows], ["AGENT_CURRENT"])
        with self.assertRaisesRegex(ValueError, "deleted"):
            repository.apply_mutations(owner_agent_id="agent", source_message_id=2,
                                       extractor_version="test", mutations=mutations, now=now)

    async def test_deleted_session_cannot_reconnect(self):
        await server.delete_session("session", self.request)
        self.app.state.server_instance_id = "server"
        socket = SimpleNamespace(
            app=self.app, query_params={"session_mode": "open", "session_id": "session"},
            accept=AsyncMock(), close=AsyncMock(),
        )
        await server.websocket_endpoint(socket)
        socket.accept.assert_awaited_once()
        socket.close.assert_awaited_once_with(code=4004, reason="Conversation deleted")


if __name__ == "__main__":
    unittest.main()
