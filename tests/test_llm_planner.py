import unittest

from app.planners.llm_planner import LLMPlanner


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def chat(self, _messages):
        return self.response


class LLMPlannerTests(unittest.TestCase):
    def test_valid_json_produces_actions(self):
        llm = FakeLLM(
            '{"actions":[{"type":"web_search","query":"python"},{"type":"respond"}]}'
        )
        planner = LLMPlanner(llm)

        plan = planner.decide("python", {})

        self.assertEqual([a.type for a in plan.actions], ["web_search", "respond"])
        self.assertEqual(plan.actions[0].payload, {"query": "python"})

    def test_extra_text_around_json_still_parses(self):
        llm = FakeLLM(
            "Sure, here is the plan:\n"
            '{"actions":[{"type":"write_memory","content":"likes tea"},{"type":"respond"}]}'
        )
        planner = LLMPlanner(llm)

        plan = planner.decide("remember this", {})

        self.assertEqual([a.type for a in plan.actions], ["write_memory", "respond"])
        self.assertEqual(plan.actions[0].payload, {"content": "likes tea"})

    def test_invalid_json_falls_back_to_respond(self):
        llm = FakeLLM("not-json")
        planner = LLMPlanner(llm)

        plan = planner.decide("hello", {})

        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].type, "respond")


if __name__ == "__main__":
    unittest.main()
