import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.assistant_state import AssistantState
from app.core.events import (
    AssistantSpeechEvent,
    UserMessageAcceptedEvent,
    AssistantTurnFailureEvent,
    AssistantThinkingEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
    AvatarAnimationEvent,
    AutonomyOutcomeEvent,
)
from app.core.orchestrator import Orchestrator
from app.core import orchestrator_factory
from app.core.conversation import InputSource, SenderAttribution, SenderType, SessionKind
from app.core.turn_input import InputModality
from app.integrations import (
    CapabilityId,
    EventId,
    EventSpec,
    IntegrationEvent,
    IntegrationRegistry,
    RuntimeIntegration,
    ToolCall,
    ToolResult,
)
from app.services.tool_executor import ToolExecutor
from app.perception.state import ImageAttachment
from app.perception.keys import PerceptionKey
from app.services.memory_action_handler import MemoryActionHandler
from app.services.memory_retriever import MemoryRetriever
from app.services.turn_finalizer import TurnFinalizer
from app.beliefs import (
    BeliefCandidateExtractor,
    BeliefRepository,
    BeliefSnapshotService,
    BeliefUpdateService,
    ConversationalBeliefObserver,
)
from app.storage.database import Database

def consume_generator(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


class FakeToolExecutor:
    def __init__(self, context: str | None = None, integration_context: str | None = None):
        self.context = context
        self.integration_context = integration_context
        self.calls = []
        self.native_contexts = []
        self.results = []
        self.close_calls = 0

    def close(self):
        self.close_calls += 1

    def get_native_tools(self, allowed_capabilities=None, **kwargs):
        self.native_contexts.append(kwargs)
        tools = [{
            "type": "function",
            "function": {
                "name": "shell__execute",
                "description": "Execute a command.",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        if allowed_capabilities is not None and CapabilityId("shell", "execute") not in allowed_capabilities:
            return []
        return tools

    def collect_context(self, session_id: str, user_text: str, max_chars: int):
        return self.integration_context[:max_chars] if self.integration_context else None

    def execute(self, call: ToolCall, session_id: str, user_text: str, approval_callback=None, **_kwargs):
        self.calls.append((call, session_id, user_text, approval_callback))
        yield AssistantStateEvent(state=AssistantState.SEARCHING)
        if self.results:
            return self.results.pop(0)
        return ToolResult.success(self.context or "tool info")

class FakeHistoryStore:
    def __init__(self):
        self.records = []
        self.recent_rows = []
        self.senders = []
        self.session_kinds = {}
        self.added_session_kinds = []

    def add(
        self, session_id: str, role: str, content: str, attachments=None,
        sender=None, session_kind=SessionKind.DIRECT,
    ):
        requested_kind = SessionKind(session_kind)
        self.session_kinds.setdefault(session_id, requested_kind)
        self.added_session_kinds.append(requested_kind)
        self.records.append((session_id, role, content, attachments or []))
        self.senders.append(sender)
        return len(self.records)

    def get_session_kind(self, session_id: str):
        return self.session_kinds.get(session_id, SessionKind.DIRECT)

    def get_recent(self, session_id: str, limit: int = 10):
        return self.recent_rows

    def get_before(self, _session_id: str, _message_id: int, limit: int = 2):
        return self.recent_rows[-limit:]

    def search_past_conversations(self, query: str, current_session: str, limit: int = 4):
        # Fake retrieved episodic memory
        return ["USER: Past question", "ASSISTANT: Past answer"]


class FakeContextBuilder:
    def __init__(self):
        self.calls = []

    def build(self, session_id: str, user_text: str, memory_context=None, integration_context=None, **kwargs):
        self.calls.append({
            "session_id": session_id,
            "user_text": user_text,
            "memory_context": memory_context,
            "integration_context": integration_context,
            "belief_context": kwargs.get("belief_context"),
            "attachments": kwargs.get("attachments", []),
            "current_sender": kwargs.get("current_sender"),
            "session_kind": kwargs.get("session_kind"),
        })
        return [{"role": "user", "content": user_text}]


class FakeBeliefContextProvider:
    def __init__(self):
        self.calls = []

    def context_for_turn(self, session_id):
        self.calls.append(session_id)
        return f"snapshot-{len(self.calls)}"

class FakeLLM:
    def __init__(self, chunks, error=None, chat_responses=None):
        self.chunks = chunks
        self.error = error
        self.calls = []
        self.chat_calls = []
        self.chat_responses = list(chat_responses or [])

    def chat(self, messages, think_override=None, options_override=None, timeout_override=None, max_retries_override=None, tools=None):
        self.chat_calls.append((messages, think_override, tools))
        if self.chat_responses:
            return self.chat_responses.pop(0)
        return {"content": ""}

    def stream_chat(self, messages, think_override=None):
        self.calls.append((messages, think_override))
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


class BufferedPhaseLLM:
    def __init__(self, responses, configured_think=True):
        self.responses = list(responses)
        self.configured_think = configured_think
        self.calls = []

    def resolve_think_value(self, override=None):
        return self.configured_think if override is None else override

    def chat_buffered(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def chat(self, **_kwargs):
        raise AssertionError("buffered ReAct path expected")

    def stream_chat(self, *_args, **_kwargs):
        raise AssertionError("streaming recovery not expected")


class FakeMemoryStore:
    def __init__(self):
        self.writes = []

    def add(self, content: str, category: str = "general", importance: int = 1):
        self.writes.append((content, category, importance))

    def get_relevant(self, query: str, limit: int = 3):
        # Fake retrieved semantic memory
        return ["User likes testing"]


class FakeSummaryStore:
    def __init__(self, existing=None):
        # If the test passes in an existing summary string, 
        # package it into the tuple format the orchestrator now expects: (summary, count)
        self.existing = (existing, 0) if existing is not None else None
        self.saved = []

    def get(self, _session_id: str):
        return self.existing

    # Add the new last_turn_count parameter
    def set(self, session_id: str, summary: str, last_turn_count: int):
        # Save the count as well so we can assert against it in tests
        self.saved.append((session_id, summary, last_turn_count))


class FakeSummarizer:
    def __init__(self, summary_text="summary"):
        self.summary_text = summary_text
        self.calls = []

    def summarize(self, messages: list[dict]):
        self.calls.append(messages)
        return self.summary_text


class FakeMemoryPolicyDecision:
    def __init__(self, content, category="general", importance=2):
        self.content = content
        self.category = category
        self.importance = importance


class FakeMemoryPolicy:
    def decide_from_action(self, action_payload: dict):
        content = action_payload.get("content")
        if not content:
            return None
        return FakeMemoryPolicyDecision(content=content)

class OrchestratorTests(unittest.TestCase):
    SESSION_ID = "session-1"

    def _build_orchestrator(
        self,
        llm_chunks=None,
        summary_existing=None,
        summary_trigger=10,
        llm_error=None,
        late_routing_enabled=False,
        chat_responses=None,
        integration_context=None,
        belief_context_provider=None,
        belief_processing_mode="disabled",
        belief_turn_preparer=None,
        database=None,
        vector_store=None,
    ):
        llm = FakeLLM(
            llm_chunks or ["Hello", " world"],
            error=llm_error,
            chat_responses=chat_responses,
        )
        history = FakeHistoryStore()
        memory = FakeMemoryStore()
        summary_store = FakeSummaryStore(existing=summary_existing)
        summarizer = FakeSummarizer()
        tool_executor = FakeToolExecutor(
            context="tool info",
            integration_context=integration_context,
        )
        context_builder = FakeContextBuilder()
        memory_policy = FakeMemoryPolicy()
        memory_retriever = MemoryRetriever(memory_store=memory, history_store=history)
        turn_finalizer = TurnFinalizer(
            history_store=history,
            summary_store=summary_store,
            summarizer=summarizer,
            summary_trigger=summary_trigger,
        )

        orch = Orchestrator(
            llm=llm,
            context_builder=context_builder,
            history_store=history,
            summary_store=summary_store,
            tool_executor=tool_executor,
            memory_retriever=memory_retriever,
            turn_finalizer=turn_finalizer,
            gesture_catalog={"greeting": "/static/animations/Gestures/Greeting.fbx"},
            late_routing_enabled=late_routing_enabled,
            belief_context_provider=belief_context_provider,
            belief_processing_mode=belief_processing_mode,
            belief_turn_preparer=belief_turn_preparer,
            database=database,
            vector_store=vector_store,
        )
        return orch, llm, history, memory, summary_store, summarizer, tool_executor, context_builder

    def test_close_releases_owned_storage_in_reverse_order_once(self):
        close_order = []
        database = SimpleNamespace(close=lambda: close_order.append("database"))
        vector_store = SimpleNamespace(close=lambda: close_order.append("vector_store"))
        built = self._build_orchestrator(
            database=database,
            vector_store=vector_store,
        )
        orchestrator, tool_executor = built[0], built[6]

        orchestrator.close()
        orchestrator.close()

        self.assertEqual(tool_executor.close_calls, 1)
        self.assertEqual(close_order, ["vector_store", "database"])

    def test_factory_closes_constructed_storage_when_build_fails(self):
        close_order = []
        database = SimpleNamespace(
            path=":memory:",
            close=lambda: close_order.append("database"),
        )
        autonomy_store = SimpleNamespace(
            close=lambda: close_order.append("autonomy_store"),
        )
        vector_store = SimpleNamespace(
            close=lambda: close_order.append("vector_store"),
        )
        config = SimpleNamespace(
            llm={"model": "test", "host": "http://localhost", "generation": {}},
            integrations={},
            local_human={"id": "person-1", "display_name": "Local Person"},
            assistant={"id": "astra", "display_name": "Astra"},
        )

        with patch.object(orchestrator_factory, "OllamaClient") as llm_type, patch.object(
            orchestrator_factory, "Database", return_value=database
        ), patch.object(
            orchestrator_factory, "AutonomyStore", return_value=autonomy_store
        ), patch.object(
            orchestrator_factory, "VectorStore", return_value=vector_store
        ), patch.object(
            orchestrator_factory,
            "ChatHistoryStore",
            side_effect=RuntimeError("history failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "history failed"):
                orchestrator_factory.build_orchestrator(config)

        llm_type.return_value.preload.assert_called_once_with()
        self.assertEqual(
            close_order,
            ["vector_store", "autonomy_store", "database"],
        )

    def test_turn_flow_injects_memory_into_context(self):
        orch, _llm, history, memory, _summary, _summarizer, tool_executor, context_builder = self._build_orchestrator(summary_trigger=999)

        list(orch.handle_user_input(self.SESSION_ID, "hello"))

        perception_snapshot = orch.perception.snapshot()
        self.assertIn(PerceptionKey.MEMORY_RETRIEVED.value, perception_snapshot)
        self.assertIn("User likes testing", perception_snapshot[PerceptionKey.MEMORY_RETRIEVED.value]["value"])

        mem_ctx = context_builder.calls[0]["memory_context"]
        tool_ctx = context_builder.calls[0]["integration_context"]
        
        self.assertIn("User likes testing", mem_ctx)
        self.assertIn("Past answer", mem_ctx)
        
        self.assertIsNone(tool_ctx)

    def test_instant_mode_skips_tool_routing_and_responds_directly(self):
        orch, _llm, _history, _memory, _summary, _summarizer, tool_executor, context_builder = self._build_orchestrator(summary_trigger=999)

        list(orch.handle_user_input(self.SESSION_ID, "hello", instant_mode=True))

        self.assertEqual(tool_executor.calls, [])
        self.assertEqual(context_builder.calls[0]["integration_context"], None)

    def test_agent_mode_uses_late_routing_chat_instead_of_streaming(self):
        (
            orch,
            llm,
            _history,
            _memory,
            _summary,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=["streaming response"],
            summary_trigger=999,
            late_routing_enabled=True,
        )

        list(orch.handle_user_input(self.SESSION_ID, "hello"))

        self.assertEqual(len(llm.chat_calls), 1)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.chat_calls[0][2][0]["function"]["name"], "shell__execute")

    def test_react_mode_propagates_persisted_authoritative_turn_unchanged(self):
        class Prepared:
            def __init__(self, turn):
                self.authoritative_turn = turn

            def tool_catalog_message(self):
                return "frozen catalog"

        class Preparer:
            def __init__(self):
                self.turns = []

            def prepare(self, turn):
                self.turns.append(turn)
                return Prepared(turn)

        preparer = Preparer()
        built = self._build_orchestrator(
            late_routing_enabled=True,
            chat_responses=[{"content": "Done"}],
            summary_trigger=999,
            belief_processing_mode="react_tool",
            belief_turn_preparer=preparer,
        )
        sender = SenderAttribution(
            "relay:human:alice", "Alice", SenderType.HUMAN,
            InputSource.MANUAL_RELAY,
        )
        list(built[0].handle_user_input(
            "group-a", "I am testing", sender=sender,
            session_kind=SessionKind.MANUAL_GROUP,
        ))

        turn = preparer.turns[0]
        forwarded = built[6].native_contexts[0]
        self.assertIs(forwarded["authoritative_turn"], turn)
        self.assertIs(forwarded["prepared_belief_turn"].authoritative_turn, turn)
        self.assertEqual(turn.session_id, "group-a")
        self.assertEqual(turn.session_kind, SessionKind.MANUAL_GROUP)
        self.assertEqual(turn.user_message_id, 1)
        self.assertEqual(turn.user_text, "I am testing")
        self.assertEqual(turn.sender_id, "relay:human:alice")
        self.assertEqual(turn.sender_display_name, "Alice")
        self.assertEqual(turn.sender_type, SenderType.HUMAN)
        self.assertEqual(turn.input_source, InputSource.MANUAL_RELAY)
        self.assertEqual(turn.owner_agent_id, "default-agent")
        self.assertEqual(turn.timezone_name, "UTC")
        self.assertIsNotNone(turn.observed_at.tzinfo)

    def test_react_instant_turn_bypasses_preparation_and_tools(self):
        class Preparer:
            def __init__(self):
                self.calls = []

            def prepare(self, turn):
                self.calls.append(turn)
                raise AssertionError("instant mode must not prepare belief tools")

        preparer = Preparer()
        built = self._build_orchestrator(
            late_routing_enabled=True,
            summary_trigger=999,
            belief_processing_mode="react_tool",
            belief_turn_preparer=preparer,
        )
        list(built[0].handle_user_input("instant", "I am busy", instant_mode=True))

        self.assertEqual(preparer.calls, [])
        self.assertEqual(built[6].native_contexts, [])
        self.assertEqual(len(built[1].calls), 1)

    def test_late_routing_prompt_calibrates_memory_and_beliefs_without_forcing_calls(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            chat_responses=[{"content": "No tool needed"}],
            summary_trigger=999,
        )
        list(built[0].handle_user_input("prompt", "hello"))
        system_text = "\n".join(
            message["content"]
            for message in built[1].chat_calls[0][0]
            if message.get("role") == "system"
        )
        self.assertIn("Use beliefs__update", system_text)
        self.assertIn("Use memory__write", system_text)
        self.assertIn("Use neither", system_text)
        self.assertIn("If no tools are needed", system_text)
        self.assertIn("at most one native tool call per inference step", system_text)

    def test_integration_context_is_injected_in_direct_and_agent_modes(self):
        direct = self._build_orchestrator(
            integration_context="connected state",
            summary_trigger=999,
        )
        list(direct[0].handle_user_input(self.SESSION_ID, "hello", instant_mode=True))
        self.assertEqual(
            direct[-1].calls[0]["integration_context"],
            "connected state",
        )

        agent = self._build_orchestrator(
            integration_context="connected state",
            late_routing_enabled=True,
            chat_responses=[{"content": "Ready"}],
            summary_trigger=999,
        )
        list(agent[0].handle_user_input(self.SESSION_ID, "hello"))
        self.assertEqual(agent[-1].calls[0]["integration_context"], "connected state")

    def test_orchestrator_collects_belief_context_for_normal_turn(self):
        provider = FakeBeliefContextProvider()
        built = self._build_orchestrator(
            summary_trigger=999,
            belief_context_provider=provider,
        )
        list(built[0].handle_user_input(self.SESSION_ID, "hello", instant_mode=True))

        self.assertEqual(provider.calls, [self.SESSION_ID])
        self.assertEqual(built[-1].calls[0]["belief_context"], "snapshot-1")

    def test_background_and_integration_turns_collect_belief_context(self):
        provider = FakeBeliefContextProvider()
        proactive = self._build_orchestrator(
            summary_trigger=999,
            belief_context_provider=provider,
        )
        list(proactive[0].handle_proactive_event(self.SESSION_ID, event_text="changed"))
        self.assertEqual(proactive[-1].calls[0]["belief_context"], "snapshot-1")
        self.assertIsNone(proactive[-1].calls[0]["current_sender"])

        integration = self._build_orchestrator(
            summary_trigger=999,
            belief_context_provider=provider,
            chat_responses=[{"content": "done"}],
        )
        event = IntegrationEvent(EventId("demo", "finished"), {}, self.SESSION_ID)
        spec = EventSpec(
            event=event.event,
            description="finished",
            payload_schema={"type": "object", "properties": {}},
        )
        list(integration[0].handle_integration_event(self.SESSION_ID, event, spec))
        self.assertEqual(integration[-1].calls[0]["belief_context"], "snapshot-2")
        self.assertIsNone(integration[-1].calls[0]["current_sender"])

    def test_group_internal_turns_never_use_local_human_attribution(self):
        proactive = self._build_orchestrator(
            summary_trigger=999,
        )
        proactive[2].session_kinds[self.SESSION_ID] = SessionKind.MANUAL_GROUP
        list(proactive[0].handle_proactive_event(self.SESSION_ID, event_text="changed"))
        proactive_sender = proactive[-1].calls[0]["current_sender"]
        self.assertEqual(proactive[-1].calls[0]["session_kind"], SessionKind.MANUAL_GROUP)
        self.assertEqual(proactive_sender.sender_type, SenderType.SYSTEM)
        self.assertEqual(proactive_sender.input_source, InputSource.SYSTEM_RUNTIME)
        self.assertNotEqual(proactive_sender.sender_id, proactive[0].local_human_id)

        integration = self._build_orchestrator(
            summary_trigger=999,
            chat_responses=[{"content": "done"}],
        )
        integration[2].session_kinds[self.SESSION_ID] = SessionKind.MANUAL_GROUP
        event = IntegrationEvent(EventId("demo", "finished"), {}, self.SESSION_ID)
        spec = EventSpec(
            event=event.event,
            description="finished",
            payload_schema={"type": "object", "properties": {}},
        )
        list(integration[0].handle_integration_event(self.SESSION_ID, event, spec))
        integration_sender = integration[-1].calls[0]["current_sender"]
        self.assertEqual(integration[-1].calls[0]["session_kind"], SessionKind.MANUAL_GROUP)
        self.assertEqual(integration_sender.sender_type, SenderType.INTEGRATION_RUNTIME)
        self.assertEqual(integration_sender.input_source, InputSource.INTEGRATION_RUNTIME)
        self.assertNotEqual(integration_sender.sender_id, integration[0].local_human_id)

    def test_handle_user_input_persists_requested_group_kind_before_first_message(self):
        built = self._build_orchestrator(
            summary_trigger=999,
        )
        list(built[0].handle_user_input(
            "new-group",
            "hello group",
            instant_mode=True,
            session_kind=SessionKind.MANUAL_GROUP,
        ))

        self.assertEqual(built[2].get_session_kind("new-group"), SessionKind.MANUAL_GROUP)
        self.assertEqual(built[2].added_session_kinds[0], SessionKind.MANUAL_GROUP)

    def test_late_routing_loop_uses_one_frozen_belief_snapshot(self):
        provider = FakeBeliefContextProvider()
        built = self._build_orchestrator(
            late_routing_enabled=True,
            summary_trigger=999,
            belief_context_provider=provider,
            chat_responses=[
                {
                    "content": "",
                    "tool_calls": [{"function": {
                        "name": "shell__execute",
                        "arguments": {"command": "pwd"},
                    }}],
                },
                {"content": "done"},
            ],
        )
        list(built[0].handle_user_input(self.SESSION_ID, "run pwd"))

        self.assertEqual(provider.calls, [self.SESSION_ID])
        self.assertEqual(built[-1].calls[0]["belief_context"], "snapshot-1")
        self.assertEqual(len(built[1].chat_calls), 2)

    def test_extractor_failure_and_malformed_output_do_not_break_real_turn(self):
        class ExtractionLLM:
            def __init__(self, response=None, error=None):
                self.response = response
                self.error = error

            def chat(self, **_kwargs):
                if self.error:
                    raise self.error
                return self.response

        cases = [
            ExtractionLLM(error=RuntimeError("extractor unavailable")),
            ExtractionLLM(response={"content": "not a tool call"}),
        ]
        for extraction_llm in cases:
            with self.subTest(error=extraction_llm.error):
                built = self._build_orchestrator(
                    summary_trigger=999,
                )
                belief_db = Database(":memory:")
                repository = BeliefRepository(belief_db)
                observer = ConversationalBeliefObserver(
                    extractor=BeliefCandidateExtractor(extraction_llm),
                    update_service=BeliefUpdateService(repository),
                    snapshot_service=BeliefSnapshotService(repository),
                    history_store=built[2],
                )
                built[0].turn_finalizer.completion_observers = [observer]

                events = list(built[0].handle_user_input(
                    self.SESSION_ID, "I'm busy for an hour", instant_mode=True
                ))

                self.assertTrue(any(
                    isinstance(event, AssistantSpeechEvent) and event.is_final
                    for event in events
                ))
                self.assertEqual(
                    repository.get_active("default-agent", self.SESSION_ID),
                    [],
                )
                repository.close()
                belief_db.conn.close()

    def test_agent_mode_executes_namespaced_tool_call_and_continues(self):
        orch, llm, _history, _memory, _summary, _summarizer, executor, _context = (
            self._build_orchestrator(
                late_routing_enabled=True,
                chat_responses=[
                    {
                        "content": "",
                        "tool_calls": [{
                            "function": {
                                "name": "shell__execute",
                                "arguments": {"command": "pwd"},
                            },
                        }],
                    },
                    {"content": "Done"},
                ],
                summary_trigger=999,
            )
        )

        list(orch.handle_user_input(self.SESSION_ID, "run pwd"))

        self.assertEqual(len(llm.chat_calls), 2)
        self.assertEqual(str(executor.calls[0][0].capability), "shell__execute")
        self.assertEqual([call[1] for call in llm.chat_calls], [True, True])

    def test_successful_belief_continuation_preserves_enabled_reasoning(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            summary_trigger=999,
        )
        client = BufferedPhaseLLM([
            self._belief_tool_response("my"),
            {"content": "Belief applied."},
        ])
        built[0].llm = client
        list(built[0].handle_user_input(
            "belief-success-phase", "my current activity is testing", think_override=True
        ))
        self.assertEqual(
            [(call["generation_phase"], call["think_override"], call["options_override"])
             for call in client.calls],
            [("initial", True, None), ("continuation", True, None)],
        )
        self.assertEqual([call["react_iteration"] for call in client.calls], [1, 2])

    def test_successful_unrelated_tool_continuation_preserves_reasoning(self):
        tool_response = {"content": "", "tool_calls": [{"function": {
            "name": "shell__execute", "arguments": {"command": "pwd"},
        }}]}
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                built = self._build_orchestrator(
                    late_routing_enabled=True,
                    summary_trigger=999,
                )
                client = BufferedPhaseLLM([tool_response, {"content": "Done."}])
                built[0].llm = client
                list(built[0].handle_user_input(
                    f"tool-phase-{enabled}", "run pwd", think_override=enabled
                ))
                self.assertEqual(
                    [(call["generation_phase"], call["think_override"])
                     for call in client.calls],
                    [("initial", enabled), ("continuation", enabled)],
                )
                self.assertTrue(all(call["options_override"] is None for call in client.calls))

    def test_unrelated_tool_then_belief_update_remain_sequential(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            chat_responses=[
                {"content": "", "tool_calls": [{"function": {
                    "name": "shell__execute", "arguments": {"command": "pwd"},
                }}]},
                {"content": "", "tool_calls": [{"function": {
                    "name": "beliefs__update",
                    "arguments": {"assertions": [], "invalidations": []},
                }}]},
                {"content": "Done"},
            ],
            summary_trigger=999,
        )
        list(built[0].handle_user_input("sequential", "run and remember"))
        self.assertEqual(len(built[1].chat_calls), 3)
        self.assertEqual(
            [str(item[0].capability) for item in built[6].calls],
            ["shell__execute", "beliefs__update"],
        )

    def test_step_exhaustion_keeps_forced_safe_final_response(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            chat_responses=[
                {"content": "", "tool_calls": [{"function": {
                    "name": "shell__execute", "arguments": {"command": "one"},
                }}]},
                {"content": "", "tool_calls": [{"function": {
                    "name": "shell__execute", "arguments": {"command": "two"},
                }}]},
            ],
            llm_chunks=["Safe final"],
            summary_trigger=999,
        )
        built[0].max_late_routing_steps = 2
        events = list(built[0].handle_user_input("exhaust", "keep trying"))
        self.assertEqual(len(built[1].chat_calls), 2)
        self.assertEqual(len(built[1].calls), 1)
        self.assertTrue(any(
            isinstance(event, AssistantSpeechEvent)
            and event.is_final
            and event.text == "Safe final"
            for event in events
        ))
        self.assertTrue(all(
            record[2].strip()
            for record in built[2].records
            if record[1] == "assistant"
        ))

    @staticmethod
    def _belief_tool_response(subject_reference="You"):
        return {"content": "", "tool_calls": [{"function": {
            "name": "beliefs__update",
            "arguments": {
                "assertions": [{
                    "subject_reference": subject_reference,
                    "predicate": "current_activity",
                    "value": "testing",
                    "visibility": "SESSION_CURRENT",
                    "expiry_policy": "END_OF_SESSION",
                    "evidence_excerpt": "my current activity is testing",
                }],
                "invalidations": [],
            },
        }}]}

    def test_belief_error_followed_by_empty_generation_forces_tool_free_response(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            chat_responses=[self._belief_tool_response(), {"content": "   "}],
            llm_chunks=["Safe recovery"],
            summary_trigger=999,
        )
        built[6].results = [ToolResult.error(
            "assertions.0.subject_reference must be copied exactly",
            diagnostics={
                "category": "subject_reference_grounding",
                "error_code": "SUBJECT_REFERENCE_GROUNDING",
                "repository_accessed": True,
            },
        )]
        events = list(built[0].handle_user_input(
            "belief-empty", "my current activity is testing"
        ))
        self.assertEqual(len(built[6].calls), 1)
        self.assertEqual(len(built[1].chat_calls), 2)
        self.assertEqual(len(built[1].calls), 1)
        self.assertTrue(any(
            isinstance(event, AssistantSpeechEvent)
            and event.is_final
            and event.text == "Safe recovery"
            for event in events
        ))
        self.assertTrue(all(
            record[2].strip()
            for record in built[2].records
            if record[1] == "assistant"
        ))

    def test_two_rejected_belief_attempts_force_response_and_bound_calls(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            chat_responses=[
                self._belief_tool_response("You"),
                self._belief_tool_response("You"),
                self._belief_tool_response("You"),
            ],
            llm_chunks=["Beliefs aside, let's continue."],
            summary_trigger=999,
        )
        built[6].results = [
            ToolResult.error("first rejection"),
            ToolResult.error("second rejection"),
        ]
        events = list(built[0].handle_user_input(
            "belief-bound", "my current activity is testing"
        ))
        self.assertEqual(len(built[6].calls), 2)
        self.assertEqual(len(built[1].chat_calls), 2)
        self.assertEqual(len(built[1].calls), 1)
        recovery_messages = built[1].calls[0][0]
        self.assertIn(
            "Do not call beliefs__update again",
            "\n".join(str(item.get("content", "")) for item in recovery_messages),
        )
        self.assertTrue(any(
            isinstance(event, AssistantSpeechEvent)
            and event.is_final
            and event.text == "Beliefs aside, let's continue."
            for event in events
        ))

    def test_two_belief_rejections_then_instant_recovery_timeout_persists_fallback(self):
        class BufferedTimeoutLLM:
            def __init__(self, responses):
                self.responses = list(responses)
                self.calls = []
                self.stream_calls = []

            def chat_buffered(self, **kwargs):
                self.calls.append(kwargs)
                if not self.responses:
                    raise TimeoutError("recovery timed out before first chunk")
                return self.responses.pop(0)

            def chat(self, **_kwargs):
                raise AssertionError("buffered ReAct path expected")

            def stream_chat(self, messages, **kwargs):
                self.stream_calls.append((messages, kwargs))
                raise TimeoutError("instant recovery timed out before first chunk")

        built = self._build_orchestrator(
            late_routing_enabled=True,
            summary_trigger=999,
        )
        client = BufferedTimeoutLLM([
            self._belief_tool_response("my"),
            self._belief_tool_response("my"),
        ])
        built[0].llm = client
        built[6].results = [
            ToolResult.error("first rejection", diagnostics={
                "category": "native_schema_validation",
                "error_code": "NATIVE_SCHEMA_VALIDATION",
                "repository_accessed": False,
            }),
            ToolResult.error("second rejection", diagnostics={
                "category": "visibility",
                "error_code": "VISIBILITY",
                "repository_accessed": True,
            }),
        ]
        events = list(built[0].handle_user_input(
            "belief-timeout", "my current activity is testing"
        ))
        final = [event.text for event in events
                 if isinstance(event, AssistantSpeechEvent) and event.is_final]
        self.assertEqual(final, [])
        failure = next(event for event in events if isinstance(event, AssistantTurnFailureEvent))
        self.assertEqual(failure.message, "Astra couldn't finish this response.")
        self.assertEqual(failure.user_message_id, 1)
        assistant_rows = [row for row in built[2].records if row[1] == "assistant"]
        self.assertEqual(assistant_rows, [])
        self.assertEqual(len(built[6].calls), 2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(client.stream_calls), 1)
        self.assertTrue(client.calls[0]["think_override"])
        self.assertIsNone(client.calls[0]["options_override"])
        self.assertFalse(client.calls[1]["think_override"])
        self.assertEqual(client.calls[1]["options_override"], {"num_predict": 512})
        self.assertEqual(
            [call["generation_phase"] for call in client.calls],
            ["initial", "correction"],
        )
        self.assertEqual([call["react_iteration"] for call in client.calls], [1, 2])
        recovery_messages, recovery_kwargs = client.stream_calls[0]
        self.assertEqual(len(recovery_messages), 2)
        self.assertEqual(recovery_messages[-1], {
            "role": "user", "content": "my current activity is testing"
        })
        self.assertIn("Latest tool observation", recovery_messages[0]["content"])
        self.assertFalse(recovery_kwargs["think_override"])
        self.assertEqual(recovery_kwargs["options_override"], {"num_predict": 192})
        self.assertEqual(recovery_kwargs["generation_deadline_s"], 180.0)
        self.assertEqual(recovery_kwargs["timeout_override"], 180.0)

    def test_late_routing_uses_extended_generation_budget(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            summary_trigger=999,
        )
        client = BufferedPhaseLLM([{"content": "Eventually answered."}])
        built[0].llm = client

        list(built[0].handle_user_input("slow-model", "take your time"))

        self.assertEqual(client.calls[0]["timeout_override"], 600.0)
        self.assertEqual(client.calls[0]["generation_deadline_s"], 600.0)

    def test_forced_recovery_empty_emits_retryable_failure_without_assistant_history(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            chat_responses=[{"content": ""}],
            llm_chunks=["   "],
            summary_trigger=999,
        )
        events = list(built[0].handle_user_input("double-empty", "hello"))
        failure = next(event for event in events if isinstance(event, AssistantTurnFailureEvent))
        self.assertEqual(failure.message, "Astra couldn't finish this response.")
        self.assertEqual(failure.user_message_id, 1)
        assistant_rows = [
            record for record in built[2].records if record[1] == "assistant"
        ]
        self.assertEqual(assistant_rows, [])

    def test_retry_reuses_existing_user_message_without_duplicate_history(self):
        built = self._build_orchestrator(
            llm_chunks=["Recovered answer"],
            summary_trigger=999,
        )
        built[2].add("retry-session", "user", "Original question")

        events = list(built[0].handle_user_input(
            "retry-session",
            "Original question",
            instant_mode=True,
            existing_user_message_id=1,
        ))

        user_rows = [row for row in built[2].records if row[1] == "user"]
        self.assertEqual(len(user_rows), 1)
        accepted = next(event for event in events if isinstance(event, UserMessageAcceptedEvent))
        self.assertEqual(accepted.message_id, 1)
        self.assertTrue(accepted.is_retry)
        self.assertTrue(any(
            isinstance(event, AssistantSpeechEvent)
            and event.is_final
            and event.text == "Recovered answer"
            for event in events
        ))

    def test_successful_corrected_belief_call_continues_to_final_response(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            chat_responses=[
                self._belief_tool_response("You"),
                self._belief_tool_response("my"),
                {"content": "Correction applied and response complete."},
            ],
            summary_trigger=999,
        )
        built[6].results = [
            ToolResult.error("use a grounded reference"),
            ToolResult.success("Belief changes applied successfully."),
        ]
        events = list(built[0].handle_user_input(
            "belief-corrected", "my current activity is testing"
        ))
        self.assertEqual(len(built[6].calls), 2)
        self.assertEqual(len(built[1].chat_calls), 3)
        self.assertEqual(
            [call[1] for call in built[1].chat_calls],
            [True, False, True],
        )
        self.assertTrue(any(
            isinstance(event, AssistantSpeechEvent)
            and event.is_final
            and event.text == "Correction applied and response complete."
            for event in events
        ))

    def test_initial_empty_no_tool_generation_forces_recovery(self):
        built = self._build_orchestrator(
            late_routing_enabled=True,
            chat_responses=[{"content": "\n\t"}],
            llm_chunks=["Recovered initial response"],
            summary_trigger=999,
        )
        events = list(built[0].handle_user_input("initial-empty", "hello"))
        self.assertEqual(len(built[1].chat_calls), 1)
        self.assertEqual(len(built[1].calls), 1)
        self.assertTrue(any(
            isinstance(event, AssistantSpeechEvent)
            and event.is_final
            and event.text == "Recovered initial response"
            for event in events
        ))

    def test_malformed_tool_name_becomes_observation_without_execution(self):
        orch, llm, history, _memory, _summary, _summarizer, executor, _context = (
            self._build_orchestrator(
                late_routing_enabled=True,
                chat_responses=[
                    {
                        "content": "",
                        "tool_calls": [{
                            "function": {"name": "shell.execute", "arguments": {}},
                        }],
                    },
                    {"content": "Recovered"},
                ],
                summary_trigger=999,
            )
        )

        list(orch.handle_user_input(self.SESSION_ID, "run something"))

        self.assertEqual(executor.calls, [])
        self.assertEqual(len(llm.chat_calls), 2)
        self.assertTrue(any(
            record[1] == "system" and "Invalid tool call" in record[2]
            for record in history.records
        ))

    def test_late_tool_execution_forwards_approval_callback(self):
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary,
            _summarizer,
            tool_executor,
            _context_builder,
        ) = self._build_orchestrator(summary_trigger=999)
        call = ToolCall(
            capability=CapabilityId("shell", "execute"),
            arguments={"command": "printf hi"},
        )

        def approve(_request):
            return True

        events, observation = consume_generator(
            orch._execute_late_tool_call(
                session_id=self.SESSION_ID,
                call=call,
                user_text="run command",
                tool_approval_callback=approve,
            )
        )

        self.assertEqual(observation.content, "tool info")
        self.assertTrue(any(isinstance(e, AssistantStateEvent) for e in events))
        self.assertEqual(
            tool_executor.calls,
            [(call, self.SESSION_ID, "run command", approve)],
        )

    def test_summarization_runs_when_threshold_reached(self):
        (
            orch,
            _llm,
            history,
            _memory,
            summary_store,
            summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(summary_trigger=2)

        history.recent_rows = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]

        list(orch.handle_user_input(self.SESSION_ID, "hello"))

        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(len(summary_store.saved), 1)
        self.assertEqual(summary_store.saved[0][1], "summary")

    def test_summarization_updates_when_summary_exists(self):
        (
            orch,
            _llm,
            history,
            _memory,
            summary_store,
            summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            summary_existing="already summarized",
            summary_trigger=1,
        )

        history.recent_rows = [{"role": "user", "content": "u1"}]

        # This will add another message to history, bringing the total to at least 2.
        # last_count is 0. Current count is 2. (2 - 0) >= 1, so it triggers.
        list(orch.handle_user_input(self.SESSION_ID, "hello"))

        # Assert that it DID summarize this time
        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(len(summary_store.saved), 1)

    def test_response_expression_tag_is_extracted_from_stream(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=["[st", "ate:happy]Hello", " there"],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        expression_values = [
            e.expression for e in events if isinstance(e, AvatarExpressionEvent)
        ]
        self.assertEqual(expression_values, ["happy"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Hello", " there"])
        self.assertEqual(speech_texts[-1], "Hello there")
        self.assertEqual(history.records[1][2], "Hello there")

    def test_response_can_switch_expressions_multiple_times(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=[
                "[state:happy]That worked. ",
                "[sta",
                "te:surprised]Wait, even better. ",
                "[state:relaxed]All set.",
            ],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        expression_values = [
            e.expression for e in events if isinstance(e, AvatarExpressionEvent)
        ]
        self.assertEqual(expression_values, ["happy", "surprised", "relaxed"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(
            speech_texts[:-1],
            ["That worked. ", "Wait, even better. ", "All set."],
        )
        self.assertEqual(speech_texts[-1], "That worked. Wait, even better. All set.")
        self.assertEqual(history.records[1][2], "That worked. Wait, even better. All set.")

    def test_response_expression_tag_allows_internal_spacing(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=["[state ", ": surprised ]Hello there"],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        expression_values = [
            e.expression for e in events if isinstance(e, AvatarExpressionEvent)
        ]
        self.assertEqual(expression_values, ["surprised"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Hello there"])
        self.assertEqual(speech_texts[-1], "Hello there")
        self.assertEqual(history.records[1][2], "Hello there")

    def test_response_expression_alias_tag_is_supported(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=["[expression:happy]Nice."],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        expression_values = [
            e.expression for e in events if isinstance(e, AvatarExpressionEvent)
        ]
        self.assertEqual(expression_values, ["happy"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Nice."])
        self.assertEqual(speech_texts[-1], "Nice.")
        self.assertEqual(history.records[1][2], "Nice.")

    def test_thinking_tags_do_not_trigger_expression_or_animation_events(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=[
                "<think>\n[state:happy][animation:greeting]secret\n</think>\n\n",
                "Hello there.",
            ],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        expression_values = [
            e.expression for e in events if isinstance(e, AvatarExpressionEvent)
        ]
        self.assertEqual(expression_values, ["neutral"])

        animation_values = [
            e.animation for e in events if isinstance(e, AvatarAnimationEvent)
        ]
        self.assertEqual(animation_values, [])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Hello there."])
        self.assertEqual(speech_texts[-1], "Hello there.")
        self.assertEqual(history.records[1][2], "Hello there.")

    def test_thinking_text_is_streamed_as_separate_events(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=[
                "<thi",
                "nk>\nStep 1",
                "\nStep 2",
                "\n</think>\n\n",
                "Answer.",
            ],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        thinking_texts = [e.text for e in events if isinstance(e, AssistantThinkingEvent)]
        self.assertEqual(thinking_texts, ["\nStep 1", "\nStep 2", "\n"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Answer."])
        self.assertEqual(speech_texts[-1], "Answer.")
        self.assertEqual(history.records[1][2], "Answer.")

    @patch("app.core.orchestrator.trace_event")
    def test_trace_logs_reasoning_response_alongside_visible_response(self, trace_event_mock):
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=[
                "<think>\nReasoning bit 1",
                "\nReasoning bit 2\n</think>\n\n",
                "Visible answer.",
            ],
            summary_trigger=999,
        )

        list(orch.handle_user_input(self.SESSION_ID, "hello"))

        llm_stream_complete_call = next(
            call for call in trace_event_mock.call_args_list
            if call.args[:2] == ("orchestrator", "llm_stream_complete")
        )

        payload = llm_stream_complete_call.kwargs["payload"]
        self.assertEqual(payload["visible_response"], "Visible answer.")
        self.assertEqual(payload["reasoning_response"], "\nReasoning bit 1\nReasoning bit 2\n")

    def test_response_animation_tag_is_extracted_from_stream(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=["Hello [animation:greeting]there."],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        animation_values = [
            e.animation for e in events if isinstance(e, AvatarAnimationEvent)
        ]
        self.assertEqual(animation_values, ["greeting"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Hello ", "there."])
        self.assertEqual(speech_texts[-1], "Hello there.")
        self.assertEqual(history.records[1][2], "Hello there.")

    def test_response_animation_alias_tag_is_supported(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=["[gesture:greeting]Hi."],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        animation_values = [
            e.animation for e in events if isinstance(e, AvatarAnimationEvent)
        ]
        self.assertEqual(animation_values, ["greeting"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Hi."])
        self.assertEqual(speech_texts[-1], "Hi.")
        self.assertEqual(history.records[1][2], "Hi.")

    def test_response_bare_animation_name_in_brackets_is_supported(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=["[greeting]Hi."],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        animation_values = [
            e.animation for e in events if isinstance(e, AvatarAnimationEvent)
        ]
        self.assertEqual(animation_values, ["greeting"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Hi."])
        self.assertEqual(speech_texts[-1], "Hi.")
        self.assertEqual(history.records[1][2], "Hi.")

    def test_unknown_animation_tag_is_stripped_without_event(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=["Hi [animation:unknown]there."],
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        animation_values = [
            e.animation for e in events if isinstance(e, AvatarAnimationEvent)
        ]
        self.assertEqual(animation_values, [])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Hi ", "there."])
        self.assertEqual(speech_texts[-1], "Hi there.")
        self.assertEqual(history.records[1][2], "Hi there.")

    def test_image_only_turn_updates_perception_history_and_context(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            context_builder,
        ) = self._build_orchestrator(summary_trigger=999)

        attachment = ImageAttachment(
            name="clipboard.png",
            mime_type="image/png",
            base64_data="aGVsbG8=",
            size_bytes=5,
        )

        list(orch.handle_user_input(self.SESSION_ID, "", attachments=[attachment]))

        self.assertEqual(context_builder.calls[0]["user_text"], "")
        perception_snapshot = orch.perception.snapshot()
        self.assertEqual(perception_snapshot["user.input"]["image_count"], 1)
        self.assertEqual(
            perception_snapshot["user.input"]["attachments"][0]["name"],
            "clipboard.png",
        )
        self.assertEqual(history.records[0][2], "[User attached 1 image]")
        self.assertEqual(history.records[0][3], [attachment])
        self.assertEqual(context_builder.calls[0]["attachments"], [attachment])

    def test_turn_can_override_reasoning_for_single_message(self):
        (
            orch,
            llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(summary_trigger=999)

        list(orch.handle_user_input(self.SESSION_ID, "hello", think_override=True))

        self.assertEqual(len(llm.calls), 1)
        self.assertIs(llm.calls[0][1], True)

    def test_proactive_event_is_hidden_system_context_not_user_history(self):
        (
            orch,
            llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            context_builder,
        ) = self._build_orchestrator(summary_trigger=999)

        attachment = ImageAttachment(
            name="screen.jpg",
            mime_type="image/jpeg",
            base64_data="aGVsbG8=",
            size_bytes=5,
        )
        event_text = "The watchdog saw a disruptive screen event."

        list(orch.handle_proactive_event(
            self.SESSION_ID,
            event_text=event_text,
            attachments=[attachment],
        ))

        self.assertEqual(context_builder.calls[0]["user_text"], "")
        self.assertEqual(context_builder.calls[0]["attachments"], [attachment])
        self.assertEqual(len(llm.calls), 1)

        messages = llm.calls[0][0]
        hidden_system_messages = [
            message
            for message in messages
            if message["role"] == "system" and event_text in message["content"]
        ]
        self.assertEqual(len(hidden_system_messages), 1)

        user_messages = [
            message
            for message in messages
            if message["role"] == "user" and event_text in message["content"]
        ]
        self.assertEqual(user_messages, [])
        self.assertEqual(len(history.records), 1)
        self.assertEqual(history.records[0][1], "assistant")
        self.assertEqual(history.records[0][2], "Hello world")

    def test_integration_event_is_silent_and_does_not_create_user_history(self):
        (
            orch, llm, history, _memory, _summary, _summarizer,
            _tool_executor, context_builder,
        ) = self._build_orchestrator(
            summary_trigger=999,
            chat_responses=[{"content": "Internal event summary."}],
        )
        event = IntegrationEvent(
            EventId("demo", "finished"), {"ok": True}, self.SESSION_ID
        )
        spec = EventSpec(
            event=event.event,
            description="Demo finished.",
            payload_schema={"type": "object", "properties": {}},
        )

        events = list(orch.handle_integration_event(self.SESSION_ID, event, spec))

        outcome = next(item for item in events if isinstance(item, AutonomyOutcomeEvent))
        self.assertEqual(outcome.summary, "Internal event summary.")
        self.assertIsNone(outcome.notification)
        self.assertFalse(any(record[1] in {"user", "assistant"} for record in history.records))
        self.assertEqual(context_builder.calls[0]["user_text"], "")
        self.assertEqual(
            {tool["function"]["name"] for tool in llm.chat_calls[0][2]},
            set(),
        )

    def test_integration_event_can_request_text_notification(self):
        (
            orch, llm, history, _memory, _summary, _summarizer,
            _fake_executor, _context_builder,
        ) = self._build_orchestrator(
            summary_trigger=999,
            chat_responses=[
                {
                    "content": "",
                    "tool_calls": [{"function": {
                        "name": "runtime__notify",
                        "arguments": {"message": "The command failed.", "delivery": "text"},
                    }}],
                },
                {"content": "Notified the user about the failure."},
            ],
        )
        orch.tool_executor = ToolExecutor(IntegrationRegistry([RuntimeIntegration()]))
        event = IntegrationEvent(
            EventId("demo", "failed"), {"ok": False}, self.SESSION_ID
        )
        spec = EventSpec(
            event=event.event,
            description="Demo failed.",
            payload_schema={"type": "object", "properties": {}},
        )

        events = list(orch.handle_integration_event(self.SESSION_ID, event, spec))

        outcome = next(item for item in events if isinstance(item, AutonomyOutcomeEvent))
        self.assertEqual(outcome.notification["message"], "The command failed.")
        self.assertIn((self.SESSION_ID, "assistant", "The command failed.", []), history.records)
        self.assertNotIn(
            (self.SESSION_ID, "assistant", "Notified the user about the failure.", []),
            history.records,
        )

    def test_turn_emits_idle_when_llm_stream_raises(self):
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            llm_chunks=["partial"],
            llm_error=RuntimeError("stream failed"),
            summary_trigger=999,
        )

        events = list(orch.handle_user_input(self.SESSION_ID, "hello"))

        state_values = [
            event.state for event in events if isinstance(event, AssistantStateEvent)
        ]
        self.assertEqual(state_values[-1], AssistantState.IDLE)
        failure = next(event for event in events if isinstance(event, AssistantTurnFailureEvent))
        self.assertEqual(failure.user_message_id, 1)

    def test_user_input_generator_close_does_not_yield_during_generatorexit(self):
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(summary_trigger=999)

        gen = orch.handle_user_input(self.SESSION_ID, "hello")
        self.assertEqual(next(gen).state, AssistantState.THINKING)
        gen.close()

    def test_proactive_generator_close_does_not_yield_during_generatorexit(self):
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(summary_trigger=999)

        gen = orch.handle_proactive_event(self.SESSION_ID)
        self.assertEqual(next(gen).state, AssistantState.THINKING)
        gen.close()

    def test_shared_orchestrator_does_not_leak_session_between_turns(self):
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _tool_executor,
            context_builder,
        ) = self._build_orchestrator(summary_trigger=999)

        list(orch.handle_user_input("session-a", "hello"))
        list(orch.handle_user_input("session-b", "hello"))

        self.assertEqual([record[0] for record in history.records], [
            "session-a",
            "session-a",
            "session-b",
            "session-b",
        ])
        self.assertEqual(context_builder.calls[0]["session_id"], "session-a")
        self.assertEqual(context_builder.calls[1]["session_id"], "session-b")

        perception_snapshot = orch.perception.snapshot()
        self.assertEqual(
            perception_snapshot[PerceptionKey.USER_INPUT.value]["text"],
            "hello",
        )

    def test_normal_text_and_voice_use_authoritative_local_identity(self):
        built = self._build_orchestrator(summary_trigger=999)
        orch, history = built[0], built[2]
        orch.local_human_id = "person-1"
        orch.local_human_name = "Local Person"

        list(orch.handle_user_input("text-session", "hello", input_modality=InputModality.TEXT))
        list(orch.handle_user_input("voice-session", "spoken", input_modality=InputModality.VOICE))

        text_sender = history.senders[0]
        voice_sender = history.senders[2]
        self.assertEqual((text_sender.sender_id, text_sender.sender_display_name), ("person-1", "Local Person"))
        self.assertEqual(text_sender.input_source.value, "local_text")
        self.assertEqual(voice_sender.input_source.value, "local_voice")


if __name__ == "__main__":
    unittest.main()
