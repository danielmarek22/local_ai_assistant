import ast
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from app.beliefs import (
    BeliefRepository,
    BeliefSnapshotService,
    BeliefTurnPreparer,
    BeliefUpdateService,
    REACT_TOOL_BELIEF_VERSION,
)
from app.core.conversation import InputSource, SenderType, SessionKind
from app.core.orchestrator_factory import _build_belief_components
from app.core.turn_completion import AuthoritativeTurnContext
from app.integrations import (
    CapabilityId,
    IntegrationRegistry,
    InvocationContext,
    ToolCall,
)
from app.integrations.beliefs import BeliefIntegration
from app.integrations.builtins import MemoryIntegration
from app.services.tool_executor import ToolExecutor
from app.storage.database import Database


NOW = datetime(2026, 8, 27, 10, 0, tzinfo=timezone.utc)


class BeliefDependencyBoundaryTests(unittest.TestCase):
    def test_belief_modules_do_not_import_integration_framework(self):
        violations = []
        for path in sorted(Path("app/beliefs").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or ""]
                else:
                    continue
                for module in imported:
                    if module == "app.integrations" or module.startswith("app.integrations."):
                        violations.append(f"{path}:{node.lineno}: {module}")
        self.assertEqual(violations, [])


class FakeHistory:
    def __init__(self):
        self.rows = []

    def get_before(self, _session_id, message_id, limit=2):
        return [row for row in self.rows if row.get("id", 0) < message_id][-limit:]

    def get_participant_senders_before(self, _session_id, message_id, limit=32):
        return [row for row in self.rows if row.get("id", 0) < message_id][-limit:]


def turn(
    message_id=1,
    text="I am testing Astra",
    *,
    sender_id="person-1",
    sender_name="Alice",
    sender_type=SenderType.HUMAN,
    input_source=InputSource.LOCAL_TEXT,
    session_id="session-a",
):
    return AuthoritativeTurnContext(
        owner_agent_id="astra",
        session_id=session_id,
        user_message_id=message_id,
        user_text=text,
        observed_at=NOW,
        timezone_name="Europe/Warsaw",
        sender_id=sender_id,
        sender_display_name=sender_name,
        sender_type=sender_type,
        input_source=input_source,
        session_kind=(
            SessionKind.MANUAL_GROUP
            if input_source == InputSource.MANUAL_RELAY
            else SessionKind.DIRECT
        ),
    )


def assertion(text="I am testing Astra", value="testing Astra"):
    return {
        "assertions": [{
            "subject_reference": "I",
            "predicate": "current_activity",
            "value": value,
            "visibility": "AGENT_CURRENT",
            "expiry_policy": "END_OF_LOCAL_DAY",
            "evidence_excerpt": text,
        }],
        "invalidations": [],
    }


class ReactBeliefToolTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.repository = BeliefRepository(self.db)
        self.snapshot = BeliefSnapshotService(self.repository, max_beliefs=24)
        self.history = FakeHistory()
        self.preparer = BeliefTurnPreparer(
            snapshot_service=self.snapshot,
            history_store=self.history,
        )
        self.integration = BeliefIntegration(
            update_service=BeliefUpdateService(
                self.repository,
                extractor_version=REACT_TOOL_BELIEF_VERSION,
            ),
            max_candidates=4,
        )
        self.registry = IntegrationRegistry([self.integration])

    def tearDown(self):
        self.repository.close()
        self.db.conn.close()

    def context(self, authoritative_turn):
        prepared = self.preparer.prepare(authoritative_turn)
        return InvocationContext(
            authoritative_turn.session_id,
            "spoofed invocation text",
            authoritative_turn=authoritative_turn,
            prepared_belief_turn=prepared,
        )

    def invoke(self, authoritative_turn, payload):
        return self.registry.invoke(
            ToolCall(CapabilityId("beliefs", "update"), payload),
            self.context(authoritative_turn),
        )

    def test_tool_is_exposed_only_with_matching_eligible_frozen_authority(self):
        eligible = turn()
        context = self.context(eligible)
        self.assertEqual(
            [item["function"]["name"] for item in self.registry.get_native_tools(
                invocation_context=context
            )],
            ["beliefs__update"],
        )
        self.assertEqual(self.registry.get_native_tools(), [])
        ungrounded = turn(text="Hello there")
        self.assertEqual(self.registry.get_native_tools(
            invocation_context=self.context(ungrounded)
        ), [])
        ineligible = turn(
            sender_type=SenderType.SYSTEM,
            input_source=InputSource.SYSTEM_RUNTIME,
        )
        self.assertEqual(self.registry.get_native_tools(
            invocation_context=InvocationContext(
                ineligible.session_id, ineligible.user_text,
                authoritative_turn=ineligible,
            )
        ), [])

    def test_eligibility_accepts_supported_participants_and_rejects_runtime_sources(self):
        eligible = (
            turn(),
            turn(input_source=InputSource.LOCAL_VOICE),
            turn(input_source=InputSource.MANUAL_RELAY),
            turn(
                sender_id="relay:external_agent:one",
                sender_name="Agent One",
                sender_type=SenderType.EXTERNAL_AGENT,
                input_source=InputSource.MANUAL_RELAY,
            ),
        )
        for authoritative in eligible:
            with self.subTest(source=authoritative.input_source):
                self.assertEqual(len(self.registry.get_native_tools(
                    invocation_context=self.context(authoritative)
                )), 1)
        ineligible = (
            (SenderType.LOCAL_ASSISTANT, InputSource.ASSISTANT_GENERATION),
            (SenderType.SYSTEM, InputSource.SYSTEM_RUNTIME),
            (SenderType.TOOL, InputSource.TOOL_RUNTIME),
            (SenderType.INTEGRATION_RUNTIME, InputSource.INTEGRATION_RUNTIME),
        )
        for sender_type, input_source in ineligible:
            authoritative = turn(sender_type=sender_type, input_source=input_source)
            with self.subTest(sender_type=sender_type):
                self.assertEqual(self.registry.get_native_tools(
                    invocation_context=InvocationContext(
                        authoritative.session_id,
                        authoritative.user_text,
                        authoritative_turn=authoritative,
                    )
                ), [])

    def test_missing_authority_and_spoof_fields_are_rejected(self):
        call = ToolCall(CapabilityId("beliefs", "update"), assertion())
        missing = self.registry.invoke(call, InvocationContext("session-a", "I am testing Astra"))
        self.assertEqual(missing.status.value, "error")
        spoofed = assertion()
        spoofed["owner_agent_id"] = "attacker"
        result = self.registry.invoke(call.__class__(call.capability, spoofed), self.context(turn()))
        self.assertEqual(result.status.value, "error")
        self.assertIn("Additional properties", result.content)

    def test_valid_assertion_uses_authoritative_text_and_records_react_version(self):
        authoritative = turn()
        result = self.invoke(authoritative, assertion())

        self.assertEqual(result.status.value, "success")
        beliefs = self.repository.get_visible("astra", "session-a", now=NOW)
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0].subject_id, "person-1")
        self.assertEqual(beliefs[0].source_sender_id, "person-1")
        self.assertEqual(beliefs[0].epistemic_status.value, "SELF_REPORT")
        self.assertTrue(self.repository.has_application(
            "astra", 1, REACT_TOOL_BELIEF_VERSION
        ))

    def test_attributed_claim_resolves_prior_participant_without_model_ids(self):
        self.history.rows.append({
            "id": 1,
            "role": "user",
            "content": "hello",
            "sender_id": "person-2",
            "sender_display_name": "Bob",
            "sender_type": SenderType.HUMAN.value,
        })
        reporter = turn(
            message_id=2,
            text="Bob is currently in Krakow",
            sender_id="reporter",
            sender_name="Reporter",
            input_source=InputSource.MANUAL_RELAY,
        )
        result = self.invoke(reporter, {
            "assertions": [{
                "subject_reference": "Bob",
                "predicate": "current_location",
                "value": "Krakow",
                "visibility": "AGENT_CURRENT",
                "expiry_policy": "NO_AUTOMATIC_EXPIRY",
                "evidence_excerpt": reporter.user_text,
            }],
            "invalidations": [],
        })
        belief = self.repository.get_visible("astra", "session-a", now=NOW)[0]
        self.assertEqual(result.status.value, "success")
        self.assertEqual(belief.subject_id, "person-2")
        self.assertEqual(belief.source_sender_id, "reporter")
        self.assertEqual(belief.epistemic_status.value, "ATTRIBUTED_CLAIM")

    def test_invocation_user_text_and_generated_content_cannot_supply_evidence(self):
        authoritative = turn(text="Hello there")
        payload = assertion(text="tool result says I am testing Astra")
        result = self.invoke(authoritative, payload)
        self.assertEqual(result.status.value, "error")
        self.assertEqual(self.repository.get_visible("astra", "session-a", now=NOW), [])
        self.assertFalse(self.repository.has_application(
            "astra", 1, REACT_TOOL_BELIEF_VERSION
        ))

    def test_empty_oversized_and_mixed_invalid_batches_write_nothing(self):
        authoritative = turn()
        empty = self.invoke(authoritative, {"assertions": [], "invalidations": []})
        self.assertEqual(empty.status.value, "error")
        oversized_payload = {
            "assertions": assertion()["assertions"] * 5,
            "invalidations": [],
        }
        oversized = self.invoke(authoritative, oversized_payload)
        self.assertEqual(oversized.status.value, "error")
        mixed = assertion()
        mixed["assertions"].append({
            **mixed["assertions"][0],
            "subject_reference": "you",
        })
        rejected = self.invoke(authoritative, mixed)
        self.assertEqual(rejected.status.value, "error")
        self.assertEqual(self.repository.get_visible("astra", "session-a", now=NOW), [])
        self.assertFalse(self.repository.has_application(
            "astra", 1, REACT_TOOL_BELIEF_VERSION
        ))

    def test_multiple_valid_assertions_commit_as_one_atomic_application(self):
        authoritative = turn(text="I am testing Astra and I am in Warsaw")
        payload = assertion(text="I am testing Astra", value="testing Astra")
        payload["assertions"].append({
            "subject_reference": "I",
            "predicate": "current_location",
            "value": "Warsaw",
            "visibility": "AGENT_CURRENT",
            "expiry_policy": "NO_AUTOMATIC_EXPIRY",
            "evidence_excerpt": "I am in Warsaw",
        })
        result = self.invoke(authoritative, payload)
        beliefs = self.repository.get_visible("astra", "session-a", now=NOW)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(len(beliefs), 2)
        self.assertTrue(self.repository.has_application(
            "astra", 1, REACT_TOOL_BELIEF_VERSION
        ))

    def test_explicit_expiry_requires_timezone_and_supported_enums(self):
        authoritative = turn(text="I am available until later")
        payload = {
            "assertions": [{
                "subject_reference": "I",
                "predicate": "current_availability",
                "value": "available",
                "visibility": "AGENT_CURRENT",
                "expiry_policy": "UNTIL_EXPLICIT_DATETIME",
                "explicit_until": "2026-08-27T15:00:00",
                "evidence_excerpt": authoritative.user_text,
            }],
            "invalidations": [],
        }
        self.assertEqual(self.invoke(authoritative, payload).status.value, "error")
        payload["assertions"][0]["visibility"] = "FOREVER"
        self.assertEqual(self.invoke(authoritative, payload).status.value, "error")

    def test_replay_and_different_second_call_do_not_add_revision_or_mutation(self):
        authoritative = turn()
        first = self.invoke(authoritative, assertion())
        replay = self.invoke(authoritative, assertion())
        different = self.invoke(
            authoritative,
            assertion(value="something forgotten"),
        )
        belief = self.repository.get_visible("astra", "session-a", now=NOW)[0]
        self.assertEqual(first.status.value, "success")
        self.assertIn("already processed", replay.content)
        self.assertIn("already processed", different.content)
        self.assertEqual(belief.revision, 1)
        self.assertEqual(belief.value, "testing Astra")

    def test_rejected_call_can_be_corrected(self):
        authoritative = turn()
        bad = assertion(text="not in message")
        self.assertEqual(self.invoke(authoritative, bad).status.value, "error")
        self.assertEqual(self.invoke(authoritative, assertion()).status.value, "success")

    def test_live_you_alias_rejection_is_precise_and_grounded_correction_applies_once(self):
        text = (
            "For this session, my current activity is calibrating Astra’s ReAct "
            "belief tool while drinking coffee."
        )
        authoritative = turn(text=text)
        prepared = self.preparer.prepare(authoritative)
        catalog = prepared.tool_catalog_message()
        self.assertIn('"subject_reference": "my"', catalog)
        self.assertNotIn('"subject_reference": "You"', catalog)
        payload = {
            "assertions": [{
                "subject_reference": "You",
                "predicate": "current_activity",
                "value": {
                    "action": "calibrating Astra's ReAct belief tool",
                    "beverage": "coffee",
                },
                "visibility": "SESSION_CURRENT",
                "expiry_policy": "END_OF_SESSION",
                "evidence_excerpt": text,
            }],
            "invalidations": [],
        }
        rejected = self.registry.invoke(
            ToolCall(CapabilityId("beliefs", "update"), payload),
            InvocationContext(
                authoritative.session_id,
                authoritative.user_text,
                authoritative_turn=authoritative,
                prepared_belief_turn=prepared,
            ),
        )
        self.assertEqual(rejected.status.value, "error")
        self.assertIn("assertions.0.subject_reference", rejected.content)
        self.assertIn("Received: 'You'", rejected.content)
        self.assertIn("Choose one of: ['my']", rejected.content)
        self.assertEqual(
            rejected.diagnostics["category"], "subject_reference_grounding"
        )
        self.assertEqual(self.repository.get_visible("astra", "session-a", now=NOW), [])
        self.assertFalse(self.repository.has_application(
            "astra", authoritative.user_message_id, REACT_TOOL_BELIEF_VERSION
        ))

        payload["assertions"][0]["subject_reference"] = "my"
        applied = self.registry.invoke(
            ToolCall(CapabilityId("beliefs", "update"), payload),
            InvocationContext(
                authoritative.session_id,
                authoritative.user_text,
                authoritative_turn=authoritative,
                prepared_belief_turn=prepared,
            ),
        )
        self.assertEqual(applied.status.value, "success")
        self.assertEqual(len(self.repository.get_visible("astra", "session-a", now=NOW)), 1)
        self.assertTrue(self.repository.has_application(
            "astra", authoritative.user_message_id, REACT_TOOL_BELIEF_VERSION
        ))
        with self.repository._connection() as connection:
            count = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM belief_applications
                WHERE owner_agent_id = ? AND source_message_id = ? AND extractor_version = ?
                """,
                ("astra", authoritative.user_message_id, REACT_TOOL_BELIEF_VERSION),
            ).fetchone()["count"]
        self.assertEqual(count, 1)

    def test_live_missing_invalidations_then_session_visibility_mismatch_is_precise(self):
        text = (
            "For this session, my current activity is calibrating Astra’s ReAct "
            "belief tool while drinking coffee."
        )
        authoritative = turn(text=text)
        prepared = self.preparer.prepare(authoritative)
        base_assertion = {
            "subject_reference": "my",
            "predicate": "is currently",
            "value": {
                "activity": "calibrating Astra's ReAct belief tool",
                "beverage": "coffee",
            },
            "visibility": "AGENT_CURRENT",
            "expiry_policy": "END_OF_SESSION",
            "evidence_excerpt": text,
        }
        context = InvocationContext(
            authoritative.session_id,
            authoritative.user_text,
            authoritative_turn=authoritative,
            prepared_belief_turn=prepared,
        )
        missing = self.registry.invoke(
            ToolCall(CapabilityId("beliefs", "update"), {
                "assertions": [base_assertion],
            }),
            context,
        )
        self.assertEqual(missing.diagnostics, {
            "category": "native_schema_validation",
            "error_code": "NATIVE_SCHEMA_VALIDATION",
            "repository_accessed": False,
        })
        self.assertIn("invalidations", missing.content)

        corrected_only_missing_field = self.registry.invoke(
            ToolCall(CapabilityId("beliefs", "update"), {
                "assertions": [base_assertion],
                "invalidations": [],
            }),
            context,
        )
        self.assertEqual(corrected_only_missing_field.diagnostics, {
            "category": "visibility",
            "error_code": "VISIBILITY",
            "repository_accessed": True,
        })
        self.assertIn("assertions.0.visibility must be SESSION_CURRENT", corrected_only_missing_field.content)
        self.assertIn("visibility='AGENT_CURRENT'", corrected_only_missing_field.content)
        self.assertEqual(self.repository.get_visible("astra", "session-a", now=NOW), [])

        valid = dict(base_assertion)
        valid["visibility"] = "SESSION_CURRENT"
        applied = self.registry.invoke(
            ToolCall(CapabilityId("beliefs", "update"), {
                "assertions": [valid],
                "invalidations": [],
            }),
            context,
        )
        self.assertEqual(applied.status.value, "success")

    def test_catalog_contains_only_source_authorized_frozen_targets(self):
        alice = turn(message_id=1, text="I am testing Astra")
        self.assertEqual(self.invoke(alice, assertion()).status.value, "success")
        bob = turn(
            message_id=2,
            text="I am reviewing Astra",
            sender_id="person-2",
            sender_name="Bob",
            input_source=InputSource.MANUAL_RELAY,
        )
        self.assertEqual(self.invoke(
            bob, assertion(text=bob.user_text, value="reviewing Astra")
        ).status.value, "success")

        alice_later = turn(message_id=3, text="I am no longer testing Astra")
        prepared = self.preparer.prepare(alice_later)
        self.assertEqual(len(prepared.permitted_invalidations), 1)
        self.assertEqual(prepared.permitted_invalidations[0].source_sender_id, "person-1")
        frozen_ids = prepared.permitted_invalidation_ids
        bob_id = next(
            belief.belief_id for belief in self.repository.get_visible(
                "astra", "session-a", now=NOW
            ) if belief.source_sender_id == "person-2"
        )
        self.assertNotIn(bob_id, frozen_ids)
        result = self.registry.invoke(
            ToolCall(CapabilityId("beliefs", "update"), {
                "assertions": [],
                "invalidations": [{
                    "target_belief_id": bob_id,
                    "evidence_excerpt": alice_later.user_text,
                }],
            }),
            InvocationContext(
                alice_later.session_id, "ignored",
                authoritative_turn=alice_later,
                prepared_belief_turn=prepared,
            ),
        )
        self.assertEqual(result.status.value, "error")
        self.assertIn("frozen permitted catalog", result.content)
        self.assertEqual(prepared.permitted_invalidation_ids, frozen_ids)

    def test_authorized_invalidation_uses_explicit_frozen_id(self):
        initial = turn()
        self.assertEqual(self.invoke(initial, assertion()).status.value, "success")
        retract = turn(message_id=2, text="I am no longer testing Astra")
        prepared = self.preparer.prepare(retract)
        belief_id = next(iter(prepared.permitted_invalidation_ids))
        result = self.registry.invoke(
            ToolCall(CapabilityId("beliefs", "update"), {
                "assertions": [],
                "invalidations": [{
                    "target_belief_id": belief_id,
                    "evidence_excerpt": retract.user_text,
                }],
            }),
            InvocationContext(
                retract.session_id, "ignored",
                authoritative_turn=retract,
                prepared_belief_turn=prepared,
            ),
        )
        self.assertEqual(result.status.value, "success")
        self.assertEqual(self.repository.get_by_id(belief_id).status, "invalidated")

    def test_tool_executor_preserves_authoritative_context_for_existing_tools(self):
        executor = ToolExecutor(self.registry)
        authoritative = turn()
        prepared = self.preparer.prepare(authoritative)
        generator = executor.execute(
            ToolCall(CapabilityId("beliefs", "update"), assertion()),
            authoritative.session_id,
            "spoofed",
            authoritative_turn=authoritative,
            prepared_belief_turn=prepared,
        )
        next(generator)
        with self.assertRaises(StopIteration) as stopped:
            next(generator)
        self.assertEqual(stopped.exception.value.status.value, "success")

    def test_belief_and_memory_descriptions_state_the_distinction(self):
        class Handler:
            def handle_payload(self, _session_id, _payload):
                return True

        registry = IntegrationRegistry([self.integration, MemoryIntegration(Handler())])
        descriptions = {
            item["function"]["name"]: item["function"]["description"]
            for item in registry.get_native_tools(invocation_context=self.context(turn()))
        }
        self.assertIn("current, revisable", descriptions["beliefs__update"])
        self.assertIn("authoritative current participant message", descriptions["beliefs__update"])
        self.assertIn("future conversation", descriptions["memory__write"])
        self.assertIn("Do not duplicate", descriptions["memory__write"])


class BeliefModeConstructionTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.history = FakeHistory()
        self.base = {
            "enabled": True,
            "processing_mode": "disabled",
            "max_existing_beliefs": 24,
            "max_snapshot_chars": 2000,
            "max_candidates": 4,
            "max_disambiguating_context_chars": 1000,
            "max_generation_tokens": 128,
            "timeout_s": 1.0,
            "max_expiry_days": 90,
        }

    def tearDown(self):
        self.db.conn.close()

    def build(self, mode, native=True):
        config = SimpleNamespace(
            local_human={"id": "person-1", "display_name": "Alice"},
            beliefs={**self.base, "processing_mode": mode},
        )
        return _build_belief_components(
            config=config,
            llm=SimpleNamespace(),
            db=self.db,
            history_store=self.history,
            agent_id="astra",
            native_late_routing_enabled=native,
        )

    def test_modes_construct_mutually_exclusive_producers(self):
        disabled = self.build("disabled")
        observer = self.build("observer")
        react = self.build("react_tool")
        self.assertEqual(disabled[2], [])
        self.assertIsNone(disabled[3])
        self.assertEqual(len(observer[2]), 1)
        self.assertIsNone(observer[3])
        self.assertEqual(react[2], [])
        self.assertIsInstance(react[3], BeliefIntegration)
        for components in (disabled, observer, react):
            components[0].close()

    def test_react_mode_requires_native_routing(self):
        with self.assertRaisesRegex(ValueError, "requires native late-routing"):
            self.build("react_tool", native=False)


if __name__ == "__main__":
    unittest.main()
