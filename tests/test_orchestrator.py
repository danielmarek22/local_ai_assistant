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
)
from app.core.orchestrator import Orchestrator
from app.core.plan import Plan
from app.perception.state import ImageAttachment
from app.perception.keys import PerceptionKey
from app.services.memory_action_handler import MemoryActionHandler
from app.services.memory_retriever import MemoryRetriever
from app.services.turn_finalizer import TurnFinalizer

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
    def __init__(self, context: str | None = None):
        self.context = context
        self.calls = []

    def execute(self, action: Action, user_text: str):
        self.calls.append((action, user_text))
        yield AssistantStateEvent(state=AssistantState.SEARCHING)
        return self.context

class FakeHistoryStore:
    def __init__(self):
        self.records = []
        self.recent_rows = []

    def add(self, session_id: str, role: str, content: str, attachments=None):
        self.records.append((session_id, role, content, attachments or []))

    def get_recent(self, session_id: str, limit: int = 10):
        return self.recent_rows

    def search_past_conversations(self, query: str, current_session: str, limit: int = 4):
        # Fake retrieved episodic memory
        return ["USER: Past question", "ASSISTANT: Past answer"]


class FakeContextBuilder:
    def __init__(self):
        self.calls = []

    def build(self, session_id: str, user_text: str, memory_context=None, tool_context=None, **kwargs):
        self.calls.append({
            "session_id": session_id,
            "user_text": user_text,
            "memory_context": memory_context,
            "tool_context": tool_context,
            "attachments": kwargs.get("attachments", []),
        })
        return [{"role": "user", "content": user_text}]

class FakeLLM:
    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error
        self.calls = []

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
    ):
        llm = FakeLLM(llm_chunks or ["Hello", " world"], error=llm_error)
        history = FakeHistoryStore()
        memory = FakeMemoryStore()
        summary_store = FakeSummaryStore(existing=summary_existing)
        summarizer = FakeSummarizer()
        planner = FakePlanner(plan=plan)
        tool_executor = FakeToolExecutor(context="tool info")
        context_builder = FakeContextBuilder()
        memory_policy = FakeMemoryPolicy()
        memory_retriever = MemoryRetriever(memory_store=memory, history_store=history)
        memory_action_handler = MemoryActionHandler(
            memory_store=memory,
            memory_policy=memory_policy,
        )
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
            planner=planner,
            tool_executor=tool_executor,
            memory_retriever=memory_retriever,
            memory_action_handler=memory_action_handler,
            turn_finalizer=turn_finalizer,
            gesture_catalog={"greeting": "/static/animations/Gestures/Greeting.fbx"},
        )
        return orch, llm, history, memory, summary_store, summarizer, planner, tool_executor, context_builder

    def test_turn_flow_injects_memory_and_tools_into_context(self):
        plan = Plan(actions=[Action(type=ActionType.WEB_SEARCH, payload={"query": "python"}), Action(type=ActionType.RESPOND)])
        orch, _llm, history, memory, _summary, _summarizer, planner, tool_executor, context_builder = self._build_orchestrator(plan=plan, summary_trigger=999)

        list(orch.handle_user_input(self.SESSION_ID, "hello"))

        # 1. Verify perception was updated with retrieved memories BEFORE planner runs
        perception_snapshot = planner.calls[0][1]
        self.assertIn(PerceptionKey.MEMORY_RETRIEVED.value, perception_snapshot)
        self.assertIn("User likes testing", perception_snapshot[PerceptionKey.MEMORY_RETRIEVED.value]["value"])

        # 2. Verify ContextBuilder received memory and tool info separately
        mem_ctx = context_builder.calls[0]["memory_context"]
        tool_ctx = context_builder.calls[0]["tool_context"]
        
        self.assertIn("User likes testing", mem_ctx)
        self.assertIn("Past answer", mem_ctx)
        
        self.assertEqual("tool info", tool_ctx)

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

        self.assertEqual(planner.calls[0][0], "user shared image attachments: clipboard.png")
        perception_snapshot = planner.calls[0][1]
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

        first_perception = planner.calls[0][1]
        second_perception = planner.calls[1][1]
        self.assertEqual(
            first_perception[PerceptionKey.USER_INPUT.value]["text"],
            "hello",
        )
        self.assertEqual(
            second_perception[PerceptionKey.USER_INPUT.value]["text"],
            "hello",
        )


if __name__ == "__main__":
    unittest.main()
