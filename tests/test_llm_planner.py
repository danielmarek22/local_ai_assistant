import unittest
import time

from app.planners.llm_planner import LLMPlanner


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def chat(self, messages, think_override=None):
        self.calls.append((messages, think_override))
        return self.response


class SlowLLM:
    def __init__(self, delay_s: float, response: str):
        self.delay_s = delay_s
        self.response = response

    def chat(self, _messages, think_override=None):
        time.sleep(self.delay_s)
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
        self.assertEqual(llm.calls[0][1], False)

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

    def test_missing_required_action_fields_are_skipped(self):
        llm = FakeLLM('{"actions":[{"type":"web_search"},{"type":"respond"}]}')
        planner = LLMPlanner(llm)

        plan = planner.decide("hello", {})

        self.assertEqual([a.type for a in plan.actions], ["respond"])

    def test_timeout_falls_back_to_respond(self):
        llm = SlowLLM(
            delay_s=0.05,
            response='{"actions":[{"type":"respond"}]}',
        )
        planner = LLMPlanner(llm, timeout_ms=10)

        plan = planner.decide("hello", {})

        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].type, "respond")


if __name__ == "__main__":
    unittest.main()
