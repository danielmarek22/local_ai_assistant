from app.core.plan import Plan


class HybridPlanner:
    def __init__(self, rule_planner, llm_planner):
        self.rule_planner = rule_planner
        self.llm_planner = llm_planner

    def decide(self, user_text: str, perception: dict) -> Plan:
        # 1. Let rules try first
        rule_plan = self.rule_planner.decide(user_text, perception)

        # 2. If rules detected a specific intent, trust them
        if self._is_confident(rule_plan):
            return rule_plan

        # 3. Otherwise, ask the LLM
        return self.llm_planner.decide(user_text, perception)

    def _is_confident(self, plan: Plan) -> bool:
        """
        Rules are confident if they did something
        more specific than a generic 'respond'.
        """
        if not plan or not plan.actions:
            return False

        # Only respond = rules didn't understand intent
        if len(plan.actions) == 1 and plan.actions[0].type == "respond":
            return False

        return True

