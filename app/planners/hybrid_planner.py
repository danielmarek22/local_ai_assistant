import logging

from app.core.plan import Plan

logger = logging.getLogger("hybrid_planner")


class HybridPlanner:
    def __init__(self, rule_planner, llm_planner):
        self.rule_planner = rule_planner
        self.llm_planner = llm_planner

    def decide(self, user_text: str, perception: dict) -> Plan:
        # 1. Let rules try first
        rule_plan = self.rule_planner.decide(user_text, perception)
        logger.debug(
            "Hybrid planner rule result: %s",
            [action.type.value for action in rule_plan.actions],
        )

        # 2. If rules detected a specific intent, trust them
        if self._is_confident(rule_plan):
            logger.info("Hybrid planner using rule-based decision")
            return rule_plan

        # 3. Otherwise, ask the LLM
        logger.info("Hybrid planner escalating to LLM planner")
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
