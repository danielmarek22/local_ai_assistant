import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app.beliefs import BeliefContextProvider, BeliefRepository, BeliefSnapshotFormatter, BeliefSnapshotService
from app.beliefs.models import (
    BeliefMutation,
    CandidateOperation,
    EpistemicStatus,
    SubjectKind,
    VisibilityPolicy,
)
from app.knowledge.models import BeliefFiltersDTO, BeliefRecordStatus
from app.knowledge.service import KnowledgeService
from app.memory.memory_store import MemoryStore
from app.services.context_builder import ContextBuilder
from app.storage.database import Database
from tests.test_server_sessions import FakeVectorStore, server_module


class FakeHistory:
    def __init__(self, sessions=("session-a", "session-b")):
        self.sessions = set(sessions)

    def session_exists(self, session_id):
        return session_id in self.sessions

    def get_recent(self, session_id, limit):
        return []


class ForbiddenDependency:
    def __init__(self, label):
        self.label = label

    def __getattr__(self, name):
        raise AssertionError(f"{self.label} must not be accessed ({name})")

    def __call__(self, *args, **kwargs):
        raise AssertionError(f"{self.label} must not be called")


class KnowledgeInspectionTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.repository = BeliefRepository(self.db)
        self.addCleanup(self.repository.close)
        self.snapshot = BeliefSnapshotService(self.repository, max_beliefs=24)
        self.provider = BeliefContextProvider(
            "agent-a", self.snapshot, BeliefSnapshotFormatter(max_chars=2000)
        )
        self.history = FakeHistory()
        self.service = KnowledgeService(
            owner_agent_id="agent-a",
            repository=self.repository,
            context_provider=self.provider,
            history_store=self.history,
        )
        self.now = datetime.now(timezone.utc)

    def add_belief(
        self,
        message_id,
        predicate,
        value,
        *,
        owner="agent-a",
        session="session-a",
        visibility=VisibilityPolicy.AGENT_CURRENT,
        expires_at=None,
        source_id="local-human",
        subject_id="local-human",
    ):
        mutation = BeliefMutation(
            operation=CandidateOperation.ASSERT,
            belief_id=None,
            visibility=visibility,
            source_session_id=session,
            subject_id=subject_id,
            subject_kind=SubjectKind.PERSON,
            subject_display_name='<img src=x onerror="alert(1)">',
            predicate=predicate,
            epistemic_status=(
                EpistemicStatus.SELF_REPORT
                if subject_id == source_id else EpistemicStatus.ATTRIBUTED_CLAIM
            ),
            source_sender_id=source_id,
            source_sender_display_name="Source <script>",
            source_sender_type="human",
            source_input_source="local_text",
            value=value,
            expires_at=expires_at,
            evidence_excerpt="Evidence <b>literal</b>",
        )
        observed_at = self.now + timedelta(microseconds=message_id)
        self.repository.apply_mutations(
            owner_agent_id=owner,
            source_message_id=message_id,
            extractor_version=f"test-{owner}-{message_id}",
            mutations=[mutation],
            now=observed_at,
        )
        return self.repository.get_visible(
            owner, session, now=self.now + timedelta(days=2)
        )

    def raw_rows(self):
        with self.repository._connection() as conn:
            return [tuple(row) for row in conn.execute(
                "SELECT * FROM beliefs ORDER BY belief_id"
            ).fetchall()]

    def test_all_records_derives_status_with_invalidation_precedence(self):
        self.add_belief(1, "active_state", "active", expires_at=self.now + timedelta(days=1))
        self.add_belief(2, "expired_state", "expired", expires_at=self.now - timedelta(days=1))
        self.add_belief(3, "invalidated_state", "invalidated", expires_at=self.now - timedelta(days=1))
        invalidated_id = self.repository.list_for_inspection(
            "agent-a", predicate="invalidated_state", now=self.now
        )[0][0]["belief_id"]
        with self.repository._connection() as conn:
            conn.execute("UPDATE beliefs SET status = 'invalidated' WHERE belief_id = ?", (invalidated_id,))
            conn.commit()

        result = self.service.list_beliefs(
            filters=BeliefFiltersDTO(), limit=50, offset=0
        )
        statuses = {record.predicate: record.record_status.value for record in result.records}
        self.assertEqual(statuses, {
            "active_state": "active",
            "expired_state": "expired",
            "invalidated_state": "invalidated",
        })

    def test_stable_pagination_counts_and_exact_filters(self):
        for index in range(5):
            self.add_belief(
                index + 1,
                f"state_{index}",
                index,
                visibility=(VisibilityPolicy.SESSION_CURRENT if index % 2 else VisibilityPolicy.AGENT_CURRENT),
                source_id=("source-a" if index < 3 else "source-b"),
                subject_id="subject-a",
            )
        first = self.service.list_beliefs(filters=BeliefFiltersDTO(), limit=2, offset=0)
        second = self.service.list_beliefs(filters=BeliefFiltersDTO(), limit=2, offset=2)
        self.assertEqual(first.total, 5)
        self.assertEqual(second.total, 5)
        self.assertEqual(len({record.belief_id for record in first.records + second.records}), 4)

        combined = self.service.list_beliefs(
            filters=BeliefFiltersDTO(
                subject_id="subject-a",
                source_sender_id="source-a",
                predicate="state_1",
                epistemic_status=EpistemicStatus.ATTRIBUTED_CLAIM,
                visibility=VisibilityPolicy.SESSION_CURRENT,
                record_status=BeliefRecordStatus.ACTIVE,
                scope_session_id="session-a",
                source_session_id="session-a",
            ),
            limit=50,
            offset=0,
        )
        self.assertEqual(combined.total, 1)
        self.assertEqual(combined.records[0].predicate, "state_1")

        for name, value in (
            ("subject_id", "subject-a"),
            ("source_sender_id", "source-b"),
            ("predicate", "state_2"),
            ("epistemic_status", EpistemicStatus.ATTRIBUTED_CLAIM),
            ("visibility", VisibilityPolicy.SESSION_CURRENT),
            ("record_status", BeliefRecordStatus.ACTIVE),
            ("scope_session_id", "session-a"),
            ("source_session_id", "session-a"),
        ):
            result = self.service.list_beliefs(
                filters=BeliefFiltersDTO(**{name: value}), limit=50, offset=0
            )
            self.assertGreater(result.total, 0, name)

    def test_owner_scoped_detail_and_safe_malformed_json(self):
        self.add_belief(1, "safe_state", {"safe": True})
        self.add_belief(2, "other_state", "secret", owner="agent-b")
        own_id = self.repository.list_for_inspection("agent-a", now=self.now)[0][0]["belief_id"]
        other_id = self.repository.list_for_inspection("agent-b", now=self.now)[0][0]["belief_id"]
        with self.repository._connection() as conn:
            conn.execute("UPDATE beliefs SET value_json = ? WHERE belief_id = ?", ("{broken", own_id))
            conn.commit()

        detail = self.service.get_belief_detail(own_id)
        self.assertIsNone(detail.value)
        self.assertEqual(detail.value_json, "{broken")
        self.assertEqual(detail.value_parse_error, "Stored value is not valid JSON.")
        self.assertIsNone(self.service.get_belief_detail(other_id))
        serialized = detail.model_dump(mode="json")
        self.assertNotIn("database_path", serialized)
        self.assertNotIn("model", serialized)
        self.assertNotIn("sql", serialized)

    def test_effective_state_uses_production_session_override(self):
        self.add_belief(1, "current_state", "global")
        self.add_belief(
            2,
            "current_state",
            "session",
            visibility=VisibilityPolicy.SESSION_CURRENT,
        )
        expected = self.snapshot.active_for_turn("agent-a", "session-a")
        response = self.service.effective_beliefs("session-a")
        self.assertEqual(
            [record.belief_id for record in response.records],
            [record.belief_id for record in expected],
        )
        self.assertEqual(response.records[0].value, "session")

    def test_empty_and_exact_production_context_preview(self):
        empty = self.service.context_preview("session-a")
        self.assertEqual(empty.state, "empty")
        self.assertEqual(empty.text, "")

        self.add_belief(1, "current_state", "visible")
        preview = self.service.context_preview("session-a")
        body = self.provider.context_for_turn("session-a")
        system = ContextBuilder("system", self.history).build(
            "session-a", "hello", belief_context=body
        )[0]["content"]
        inserted_section = next(
            section for section in system.split("\n\n---\n\n")
            if section.startswith("CURRENT BELIEF STATE")
        )
        self.assertEqual(preview.state, "formatted")
        self.assertEqual(preview.text, inserted_section)

    def test_inspection_is_read_only_and_independent_of_extraction(self):
        self.add_belief(1, "stored_state", "still visible")
        before = self.raw_rows()
        self.service.list_beliefs(filters=BeliefFiltersDTO(), limit=50, offset=0)
        self.service.effective_beliefs("session-a")
        self.service.context_preview("session-a")
        detail_id = self.repository.list_for_inspection("agent-a", now=self.now)[0][0]["belief_id"]
        self.service.get_belief_detail(detail_id)
        self.assertEqual(self.raw_rows(), before)
        self.assertEqual(self.provider.context_for_turn("session-a"), self.service.context_provider.context_for_turn("session-a"))


class KnowledgeRouteTests(unittest.TestCase):
    def setUp(self):
        db = Database(":memory:")
        self.db = db
        self.repository = BeliefRepository(db)
        self.addCleanup(self.repository.close)
        provider = BeliefContextProvider(
            "agent-a",
            BeliefSnapshotService(self.repository),
            BeliefSnapshotFormatter(),
        )
        self.orchestrator = SimpleNamespace(
            agent_id="agent-a",
            belief_repository=self.repository,
            belief_context_provider=provider,
            history=FakeHistory(),
            memory_retriever=SimpleNamespace(memory=MemoryStore(db, FakeVectorStore())),
            model=ForbiddenDependency("model"),
            memory_tool=ForbiddenDependency("memory tool"),
        )
        server_module.app.state.orchestrator = self.orchestrator

    def request(self, method, path, **kwargs):
        async def run_request():
            transport = httpx.ASGITransport(app=server_module.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)
        return asyncio.run(run_request())

    def test_successful_route_dtos_and_exact_preview(self):
        now = datetime.now(timezone.utc)
        self.repository.apply_mutations(
            owner_agent_id="agent-a",
            source_message_id=1,
            extractor_version="route-test",
            mutations=[BeliefMutation(
                operation=CandidateOperation.ASSERT,
                belief_id=None,
                visibility=VisibilityPolicy.AGENT_CURRENT,
                source_session_id="session-a",
                subject_id="local-human",
                subject_kind=SubjectKind.PERSON,
                subject_display_name="You",
                predicate="route_state",
                epistemic_status=EpistemicStatus.SELF_REPORT,
                source_sender_id="local-human",
                source_sender_display_name="You",
                source_sender_type="human",
                source_input_source="local_text",
                value={"route": "visible"},
                evidence_excerpt="route evidence",
            )],
            now=now,
        )
        listed = self.request("GET", "/api/knowledge/beliefs")
        self.assertEqual(listed.status_code, 200)
        payload = listed.json()
        self.assertEqual(payload["total"], 1)
        belief_id = payload["records"][0]["belief_id"]
        detail = self.request("GET", f"/api/knowledge/beliefs/{belief_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["value"], {"route": "visible"})
        self.assertTrue({"database_path", "sql", "model"}.isdisjoint(detail.json()))

        effective = self.request(
            "GET", "/api/knowledge/beliefs/effective", params={"session_id": "session-a"}
        )
        context = self.request(
            "GET", "/api/knowledge/belief-context", params={"session_id": "session-a"}
        )
        self.assertEqual(effective.status_code, 200)
        self.assertEqual(effective.json()["records"][0]["belief_id"], belief_id)
        self.assertEqual(context.status_code, 200)
        self.assertTrue(context.json()["text"].startswith("CURRENT BELIEF STATE"))

    def test_saved_memory_route_returns_every_safe_scalar_without_mutation(self):
        rows = [
            ("saved-a", "general", "Older memory", 1, "2026-08-25 09:00:00", "2026-08-25 10:00:00"),
            ("saved-b", "preference", '<script>alert("x")</script>', 4, "2026-08-26 09:00:00", "2026-08-26 10:00:00"),
        ]
        self.db.conn.executemany(
            """
            INSERT INTO memory (id, category, content, importance, created_at, last_accessed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.db.conn.commit()
        before = [tuple(row) for row in self.db.conn.execute(
            "SELECT * FROM memory ORDER BY id"
        ).fetchall()]
        collection = self.orchestrator.memory_retriever.memory.collection
        with patch.object(
            collection,
            "query",
            side_effect=AssertionError("vector retrieval must not run"),
        ), patch.object(
            collection,
            "add",
            side_effect=AssertionError("embeddings must not be created"),
        ), patch.object(
            self.orchestrator.memory_retriever.memory,
            "get_relevant",
            side_effect=AssertionError("production retrieval must not run"),
        ), patch.object(
            self.orchestrator.memory_retriever.memory,
            "add",
            side_effect=AssertionError("memory-tool writes must not run"),
        ):
            response = self.request("GET", "/api/knowledge/memories")
        after = [tuple(row) for row in self.db.conn.execute(
            "SELECT * FROM memory ORDER BY id"
        ).fetchall()]
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual([record["id"] for record in payload["records"]], ["saved-b", "saved-a"])
        self.assertEqual(
            set(payload["records"][0]),
            {"id", "category", "content", "importance", "created_at", "last_accessed_at"},
        )
        self.assertEqual(before, after)
        self.assertTrue({"database_path", "sql", "embeddings", "configuration"}.isdisjoint(payload))

    def test_saved_memory_route_empty_and_unavailable(self):
        response = self.request("GET", "/api/knowledge/memories")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"records": [], "total": 0})
        self.orchestrator.memory_retriever = None
        self.assertEqual(self.request("GET", "/api/knowledge/memories").status_code, 503)

    def test_unknown_session_belief_and_unavailable_subsystem(self):
        self.assertEqual(
            self.request("GET", "/api/knowledge/beliefs/effective", params={"session_id": "missing"}).status_code,
            404,
        )
        self.assertEqual(self.request("GET", "/api/knowledge/beliefs/missing").status_code, 404)
        self.assertEqual(
            self.request("GET", "/api/knowledge/belief-context", params={"session_id": "missing"}).status_code,
            404,
        )
        self.assertEqual(
            self.request("GET", "/api/knowledge/beliefs/effective", params={"session_id": "session-b"}).status_code,
            200,
        )
        self.orchestrator.belief_repository = None
        self.assertEqual(self.request("GET", "/api/knowledge/beliefs").status_code, 503)

    def test_query_validation_bounds_and_enums(self):
        cases = [
            {"limit": 101},
            {"limit": 0},
            {"offset": -1},
            {"offset": 100001},
            {"record_status": "deleted"},
            {"visibility": "private"},
            {"epistemic_status": "unknown"},
            {"predicate": "x" * 65},
            {"subject_id": "x" * 129},
        ]
        for params in cases:
            with self.subTest(params=params):
                self.assertEqual(self.request("GET", "/api/knowledge/beliefs", params=params).status_code, 422)
        self.assertEqual(
            self.request(
                "GET",
                "/api/knowledge/beliefs/effective",
                params={"session_id": "bad\nvalue"},
            ).status_code,
            422,
        )


if __name__ == "__main__":
    unittest.main()
