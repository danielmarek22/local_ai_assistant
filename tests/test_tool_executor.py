import unittest

from app.core.actions import Action, ActionType
from app.core.assistant_state import AssistantState
from app.core.events import AssistantStateEvent
from app.services.tool_executor import ToolExecutor


def consume_generator(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


class FakeTool:
    def __init__(self, is_available=True, result="ctx", raises=False):
        self.is_available = is_available
        self.result = result
        self.raises = raises
        self.queries = []

    def run(self, query: str):
        self.queries.append(query)
        if self.raises:
            raise RuntimeError("boom")
        return self.result


class ToolExecutorTests(unittest.TestCase):
    def test_unregistered_tool_returns_none_without_events(self):
        executor = ToolExecutor(tools={})
        action = Action(type=ActionType.WEB_SEARCH, payload={"query": "q"})

        events, result = consume_generator(executor.execute(action, "user text"))

        self.assertEqual(events, [])
        self.assertIsNone(result)

    def test_unavailable_tool_returns_none_without_events(self):
        tool = FakeTool(is_available=False)
        executor = ToolExecutor(tools={"web_search": tool})
        action = Action(type=ActionType.WEB_SEARCH, payload={"query": "q"})

        events, result = consume_generator(executor.execute(action, "user text"))

        self.assertEqual(events, [])
        self.assertIsNone(result)

    def test_available_tool_yields_search_state_and_returns_context(self):
        tool = FakeTool(is_available=True, result="context")
        executor = ToolExecutor(tools={"web_search": tool})
        action = Action(type=ActionType.WEB_SEARCH, payload={"query": "my query"})

        events, result = consume_generator(executor.execute(action, "user text"))

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], AssistantStateEvent)
        self.assertEqual(events[0].state, AssistantState.SEARCHING)
        self.assertEqual(result, "context")
        self.assertEqual(tool.queries, ["my query"])

    def test_uses_user_text_when_query_missing(self):
        tool = FakeTool(is_available=True, result="context")
        executor = ToolExecutor(tools={"web_search": tool})
        action = Action(type=ActionType.WEB_SEARCH, payload={})

        _events, _result = consume_generator(executor.execute(action, "fallback query"))

        self.assertEqual(tool.queries, ["fallback query"])

    def test_tool_error_returns_none_after_search_state(self):
        tool = FakeTool(is_available=True, raises=True)
        executor = ToolExecutor(tools={"web_search": tool})
        action = Action(type=ActionType.WEB_SEARCH, payload={"query": "q"})

        events, result = consume_generator(executor.execute(action, "user text"))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].state, AssistantState.SEARCHING)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()