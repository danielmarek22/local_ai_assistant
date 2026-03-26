import unittest

from app.core.actions import ActionType
from app.planners.rule_planner import Planner


class RulePlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = Planner()

    def test_memory_command_creates_write_and_respond_actions(self):
        plan = self.planner.decide("Remember that my birthday is July 8", {})

        self.assertEqual(len(plan.actions), 2)
        self.assertEqual(plan.actions[0].type, ActionType.WRITE_MEMORY)
        self.assertEqual(
            plan.actions[0].payload,
            {"content": "my birthday is July 8"},
        )
        self.assertEqual(plan.actions[1].type, ActionType.RESPOND)

    def test_search_query_creates_web_search_then_respond(self):
        plan = self.planner.decide("What is the latest news about Rust?", {})

        self.assertEqual(len(plan.actions), 2)
        self.assertEqual(plan.actions[0].type, ActionType.WEB_SEARCH)
        self.assertEqual(
            plan.actions[0].payload,
            {"query": "What is the latest news about Rust?"},
        )
        self.assertEqual(plan.actions[1].type, ActionType.RESPOND)

    def test_default_falls_back_to_respond_only(self):
        plan = self.planner.decide("Tell me a joke.", {})

        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].type, ActionType.RESPOND)

    def test_empty_memory_command_falls_back_to_respond(self):
        # "Remember this" with no trailing context used to crash or save empty strings
        plan = self.planner.decide("Remember this", {})

        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].type, ActionType.RESPOND)


if __name__ == "__main__":
    unittest.main()