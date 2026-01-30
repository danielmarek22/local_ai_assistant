from app.core.planner import Planner, PlannerDecision
from app.core.llm_planner import LLMPlanner


class HybridPlanner:
    def __init__(self, rule_planner: Planner, llm_planner: LLMPlanner):
        self.rule_planner = rule_planner
        self.llm_planner = llm_planner

    def decide(self, user_text: str):
        # 1. Ask LLM planner first
        llm_decision = self.llm_planner.decide(user_text)

        if llm_decision:
            return llm_decision

        # 2. Fallback to rule-based planner
        rule_decision = self.rule_planner.decide(user_text)

        if rule_decision.use_web:
            return PlannerDecision(
                action="web_search",
                query=rule_decision.query,
                reason="rule_fallback"
            )

        return PlannerDecision(
            action="respond",
            reason="rule_fallback"
        )
