from app.planners.rule_planner import Planner
from app.planners.llm_planner import LLMPlanner
from app.planners.hybrid_planner import HybridPlanner


def build_planner(config, llm):
    mode = config.planner.get("mode", "rule")
    llm_enabled = config.planner.get("llm_enabled", False)

    rule_planner = Planner()

    if mode == "rule" or not llm_enabled:
        return rule_planner

    llm_planner = LLMPlanner(llm)

    if mode == "llm":
        return llm_planner

    if mode == "hybrid":
        return HybridPlanner(
            rule_planner=rule_planner,
            llm_planner=llm_planner,
        )

    raise ValueError(f"Unknown planner mode: {mode}")
