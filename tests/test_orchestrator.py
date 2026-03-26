import unittest

from app.core.actions import Action
from app.core.assistant_state import AssistantState
from app.core.events import (
    AssistantSpeechEvent,
    AssistantStateEvent,
    AvatarExpressionEvent,
)
from app.core.orchestrator import Orchestrator
from app.core.plan import Plan
from app.perception.state import ImageAttachment


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

    def build(self, session_id: str, user_text: str, tool_context=None):
        self.calls.append((session_id, user_text, tool_context))
        return [{"role": "user", "content": user_text}]


class FakeLLM:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def stream_chat(self, messages, think_override=None):
        self.calls.append((messages, think_override))
        for chunk in self.chunks:
            yield chunk


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
        self.existing = existing
        self.saved = []

    def get(self, _session_id: str):
        return self.existing

    def set(self, session_id: str, summary: str):
        self.saved.append((session_id, summary))


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
    
class FakeContextBuilder:
    def __init__(self):
        self.calls = []

    def build(self, session_id: str, user_text: str, injected_context=None, **kwargs):
        self.calls.append({
            "session_id": session_id,
            "user_text": user_text,
            "injected_context": injected_context,
            "attachments": kwargs.get("attachments", []),
        })
        return [{"role": "user", "content": user_text}]


class OrchestratorTests(unittest.TestCase):
    def _build_orchestrator(self, plan: Plan, llm_chunks=None, summary_existing=None, summary_trigger=10):
        llm = FakeLLM(llm_chunks or ["Hello", " world"])
        history = FakeHistoryStore()
        memory = FakeMemoryStore()
        summary_store = FakeSummaryStore(existing=summary_existing)
        summarizer = FakeSummarizer()
        planner = FakePlanner(plan=plan)
        tool_executor = FakeToolExecutor(context="tool info")
        context_builder = FakeContextBuilder()
        memory_policy = FakeMemoryPolicy()

        orch = Orchestrator(
            llm=llm,
            context_builder=context_builder,
            history_store=history,
            memory_store=memory,
            summary_store=summary_store,
            summarizer=summarizer,
            planner=planner,
            memory_policy=memory_policy,
            tool_executor=tool_executor,
            summary_trigger=summary_trigger,
        )
        return orch, llm, history, memory, summary_store, summarizer, planner, tool_executor, context_builder

    def test_turn_flow_injects_memory_and_tools_into_context(self):
        plan = Plan(actions=[Action(type="web_search", payload={"query": "python"}), Action(type="respond")])
        orch, _llm, history, memory, _summary, _summarizer, planner, tool_executor, context_builder = self._build_orchestrator(plan=plan, summary_trigger=999)

        list(orch.handle_user_input("hello"))

    # 1. Verify perception was updated with retrieved memories BEFORE planner runs
        perception_snapshot = planner.calls[0][1]
        self.assertIn("memory.retrieved", perception_snapshot)
        self.assertIn("User likes testing", perception_snapshot["memory.retrieved"].value["value"])

        # 2. Verify ContextBuilder received BOTH memory and tool info in injected_context
        injected_context = context_builder.calls[0]["injected_context"]
        self.assertIn("RETRIEVED MEMORY", injected_context)
        self.assertIn("User likes testing", injected_context)
        self.assertIn("Past answer", injected_context)
        self.assertIn("TOOL RESULTS", injected_context)
        self.assertIn("tool info", injected_context)

    def test_summarization_runs_when_threshold_reached(self):
        plan = Plan(actions=[Action(type="respond")])
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

        list(orch.handle_user_input("hello"))

        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(len(summary_store.saved), 1)
        self.assertEqual(summary_store.saved[0][1], "summary")

    def test_summarization_skips_when_summary_exists(self):
        plan = Plan(actions=[Action(type="respond")])
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

        list(orch.handle_user_input("hello"))

        self.assertEqual(len(summarizer.calls), 0)
        self.assertEqual(len(summary_store.saved), 0)

    def test_set_session_updates_session_id_and_resets_perception(self):
        plan = Plan(actions=[Action(type="respond")])
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

        original_perception = orch.perception
        orch.perception.update("user.input", {"text": "hello"})

        orch.set_session("session-2")

        self.assertEqual(orch.session_id, "session-2")
        self.assertIsNot(orch.perception, original_perception)
        self.assertEqual(orch.perception.snapshot(), {})

    def test_response_expression_tag_is_extracted_from_stream(self):
        plan = Plan(actions=[Action(type="respond")])
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

        events = list(orch.handle_user_input("hello"))

        expression_values = [
            e.expression for e in events if isinstance(e, AvatarExpressionEvent)
        ]
        self.assertEqual(expression_values, ["happy"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Hello", " there"])
        self.assertEqual(speech_texts[-1], "Hello there")
        self.assertEqual(history.records[1][2], "Hello there")

    def test_response_can_switch_expressions_multiple_times(self):
        plan = Plan(actions=[Action(type="respond")])
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

        events = list(orch.handle_user_input("hello"))

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
        plan = Plan(actions=[Action(type="respond")])
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

        events = list(orch.handle_user_input("hello"))

        expression_values = [
            e.expression for e in events if isinstance(e, AvatarExpressionEvent)
        ]
        self.assertEqual(expression_values, ["surprised"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Hello there"])
        self.assertEqual(speech_texts[-1], "Hello there")
        self.assertEqual(history.records[1][2], "Hello there")

    def test_response_expression_alias_tag_is_supported(self):
        plan = Plan(actions=[Action(type="respond")])
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

        events = list(orch.handle_user_input("hello"))

        expression_values = [
            e.expression for e in events if isinstance(e, AvatarExpressionEvent)
        ]
        self.assertEqual(expression_values, ["happy"])

        speech_texts = [e.text for e in events if isinstance(e, AssistantSpeechEvent)]
        self.assertEqual(speech_texts[:-1], ["Nice."])
        self.assertEqual(speech_texts[-1], "Nice.")
        self.assertEqual(history.records[1][2], "Nice.")

    def test_image_only_turn_updates_perception_history_and_context(self):
        plan = Plan(actions=[Action(type="respond")])
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

        list(orch.handle_user_input("", attachments=[attachment]))

        self.assertEqual(planner.calls[0][0], "user shared image attachments: clipboard.png")
        perception_snapshot = planner.calls[0][1]
        self.assertEqual(perception_snapshot["user.input"].value["image_count"], 1)
        self.assertEqual(
            perception_snapshot["user.input"].value["attachments"][0]["name"],
            "clipboard.png",
        )
        self.assertEqual(history.records[0][2], "[User attached 1 image]")
        self.assertEqual(history.records[0][3], [attachment])
        self.assertEqual(context_builder.calls[0]["attachments"], [attachment])

    def test_turn_can_override_reasoning_for_single_message(self):
        plan = Plan(actions=[Action(type="respond")])
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

        list(orch.handle_user_input("hello", think_override=True))

        self.assertEqual(len(llm.calls), 1)
        self.assertIs(llm.calls[0][1], True)


if __name__ == "__main__":
    unittest.main()
