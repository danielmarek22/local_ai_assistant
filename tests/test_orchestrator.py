import unittest
from unittest.mock import patch

from app.core.actions import Action, ActionType
from app.core.assistant_state import AssistantState
from app.core.events import (
    AssistantSpeechEvent,
    AssistantThinkingEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
    AvatarAnimationEvent,
    AutonomyOutcomeEvent,
)
from app.core.orchestrator import Orchestrator
from app.core.conversation import InputSource, SenderType, SessionKind
from app.core.turn_input import InputModality
from app.core.plan import Plan
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


class FakePlanner:
    def __init__(self, plan: Plan):
        self.plan = plan
        self.calls = []

    def decide(self, user_text: str, perception: dict) -> Plan:
        self.calls.append((user_text, perception))
        return self.plan


class FakeToolExecutor:
    def __init__(self, context: str | None = None, integration_context: str | None = None):
        self.context = context
        self.integration_context = integration_context
        self.calls = []

    def get_native_tools(self, allowed_capabilities=None):
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
        plan: Plan,
        llm_chunks=None,
        summary_existing=None,
        summary_trigger=10,
        llm_error=None,
        late_routing_enabled=False,
        chat_responses=None,
        integration_context=None,
        belief_context_provider=None,
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
        planner = FakePlanner(plan=plan)
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
        )
        return orch, llm, history, memory, summary_store, summarizer, planner, tool_executor, context_builder

    def test_turn_flow_injects_memory_into_context(self):
        plan = Plan(actions=[Action(type=ActionType.WEB_SEARCH, payload={"query": "python"}), Action(type=ActionType.RESPOND)])
        orch, _llm, history, memory, _summary, _summarizer, planner, tool_executor, context_builder = self._build_orchestrator(plan=plan, summary_trigger=999)

        list(orch.handle_user_input(self.SESSION_ID, "hello"))

        perception_snapshot = orch.perception.snapshot()
        self.assertIn(PerceptionKey.MEMORY_RETRIEVED.value, perception_snapshot)
        self.assertIn("User likes testing", perception_snapshot[PerceptionKey.MEMORY_RETRIEVED.value]["value"])

        mem_ctx = context_builder.calls[0]["memory_context"]
        tool_ctx = context_builder.calls[0]["integration_context"]
        
        self.assertIn("User likes testing", mem_ctx)
        self.assertIn("Past answer", mem_ctx)
        
        self.assertIsNone(tool_ctx)

    def test_instant_mode_skips_planner_and_responds_directly(self):
        plan = Plan(actions=[
            Action(type=ActionType.WEB_SEARCH, payload={"query": "python"}),
            Action(type=ActionType.RESPOND),
        ])
        orch, _llm, _history, _memory, _summary, _summarizer, planner, tool_executor, context_builder = self._build_orchestrator(plan=plan, summary_trigger=999)

        list(orch.handle_user_input(self.SESSION_ID, "hello", instant_mode=True))

        self.assertEqual(planner.calls, [])
        self.assertEqual(tool_executor.calls, [])
        self.assertEqual(context_builder.calls[0]["integration_context"], None)

    def test_agent_mode_uses_late_routing_chat_instead_of_streaming(self):
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            llm,
            _history,
            _memory,
            _summary,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
            llm_chunks=["streaming response"],
            summary_trigger=999,
            late_routing_enabled=True,
        )

        list(orch.handle_user_input(self.SESSION_ID, "hello"))

        self.assertEqual(len(llm.chat_calls), 1)
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(llm.chat_calls[0][2][0]["function"]["name"], "shell__execute")

    def test_integration_context_is_injected_in_direct_and_agent_modes(self):
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        direct = self._build_orchestrator(
            plan=plan,
            integration_context="connected state",
            summary_trigger=999,
        )
        list(direct[0].handle_user_input(self.SESSION_ID, "hello", instant_mode=True))
        self.assertEqual(
            direct[-1].calls[0]["integration_context"],
            "connected state",
        )

        agent = self._build_orchestrator(
            plan=plan,
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
            plan=Plan(actions=[Action(type=ActionType.RESPOND)]),
            summary_trigger=999,
            belief_context_provider=provider,
        )
        list(built[0].handle_user_input(self.SESSION_ID, "hello", instant_mode=True))

        self.assertEqual(provider.calls, [self.SESSION_ID])
        self.assertEqual(built[-1].calls[0]["belief_context"], "snapshot-1")

    def test_background_and_integration_turns_collect_belief_context(self):
        provider = FakeBeliefContextProvider()
        proactive = self._build_orchestrator(
            plan=Plan(actions=[Action(type=ActionType.RESPOND)]),
            summary_trigger=999,
            belief_context_provider=provider,
        )
        list(proactive[0].handle_proactive_event(self.SESSION_ID, event_text="changed"))
        self.assertEqual(proactive[-1].calls[0]["belief_context"], "snapshot-1")
        self.assertIsNone(proactive[-1].calls[0]["current_sender"])

        integration = self._build_orchestrator(
            plan=Plan(actions=[Action(type=ActionType.RESPOND)]),
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
            plan=Plan(actions=[Action(type=ActionType.RESPOND)]),
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
            plan=Plan(actions=[Action(type=ActionType.RESPOND)]),
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
            plan=Plan(actions=[Action(type=ActionType.RESPOND)]),
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
            plan=Plan(actions=[Action(type=ActionType.RESPOND)]),
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
                    plan=Plan(actions=[Action(type=ActionType.RESPOND)]),
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        orch, llm, _history, _memory, _summary, _summarizer, _planner, executor, _context = (
            self._build_orchestrator(
                plan=plan,
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

    def test_malformed_tool_name_becomes_observation_without_execution(self):
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        orch, llm, history, _memory, _summary, _summarizer, _planner, executor, _context = (
            self._build_orchestrator(
                plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary,
            _summarizer,
            _planner,
            tool_executor,
            _context_builder,
        ) = self._build_orchestrator(plan=plan, summary_trigger=999)
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            summary_store,
            summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(plan=plan, summary_trigger=2)

        history.recent_rows = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]

        list(orch.handle_user_input(self.SESSION_ID, "hello"))

        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(len(summary_store.saved), 1)
        self.assertEqual(summary_store.saved[0][1], "summary")

    def test_summarization_updates_when_summary_exists(self):
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            summary_store,
            summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            planner,
            _tool_executor,
            context_builder,
        ) = self._build_orchestrator(plan=plan, summary_trigger=999)

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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(plan=plan, summary_trigger=999)

        list(orch.handle_user_input(self.SESSION_ID, "hello", think_override=True))

        self.assertEqual(len(llm.calls), 1)
        self.assertIs(llm.calls[0][1], True)

    def test_proactive_event_is_hidden_system_context_not_user_history(self):
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            planner,
            _tool_executor,
            context_builder,
        ) = self._build_orchestrator(plan=plan, summary_trigger=999)

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

        self.assertEqual(planner.calls, [])
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch, llm, history, _memory, _summary, _summarizer, _planner,
            _tool_executor, context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch, llm, history, _memory, _summary, _summarizer, _planner,
            _fake_executor, _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(
            plan=plan,
            llm_chunks=["partial"],
            llm_error=RuntimeError("stream failed"),
            summary_trigger=999,
        )

        events = []
        with self.assertRaises(RuntimeError):
            for event in orch.handle_user_input(self.SESSION_ID, "hello"):
                events.append(event)

        state_values = [
            event.state for event in events if isinstance(event, AssistantStateEvent)
        ]
        self.assertEqual(state_values[-1], AssistantState.IDLE)

    def test_user_input_generator_close_does_not_yield_during_generatorexit(self):
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(plan=plan, summary_trigger=999)

        gen = orch.handle_user_input(self.SESSION_ID, "hello")
        self.assertEqual(next(gen).state, AssistantState.THINKING)
        gen.close()

    def test_proactive_generator_close_does_not_yield_during_generatorexit(self):
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            _history,
            _memory,
            _summary_store,
            _summarizer,
            _planner,
            _tool_executor,
            _context_builder,
        ) = self._build_orchestrator(plan=plan, summary_trigger=999)

        gen = orch.handle_proactive_event(self.SESSION_ID)
        self.assertEqual(next(gen).state, AssistantState.THINKING)
        gen.close()

    def test_shared_orchestrator_does_not_leak_session_between_turns(self):
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        (
            orch,
            _llm,
            history,
            _memory,
            _summary_store,
            _summarizer,
            planner,
            _tool_executor,
            context_builder,
        ) = self._build_orchestrator(plan=plan, summary_trigger=999)

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
        plan = Plan(actions=[Action(type=ActionType.RESPOND)])
        built = self._build_orchestrator(plan=plan, summary_trigger=999)
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
