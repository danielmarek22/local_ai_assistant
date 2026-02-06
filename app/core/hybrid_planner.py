from app.core.plan import Plan


class HybridPlanner:
    def __init__(self, rule_planner, llm_planner):
        self.rule_planner = rule_planner
        self.llm_planner = llm_planner

    def decide(self, user_text: str) -> Plan:
        # 1. Try LLM planner
        try:
            plan = self.llm_planner.decide(user_text)
            if plan and plan.actions:
                return plan
        except Exception:
            pass

        # 2. Fallback to rule-based planner
        return self.rule_planner.decide(user_text)
