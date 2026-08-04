
# Planners

Legacy planning components.

The active turn architecture now uses late routing in `app/core/orchestrator.py`.
The model decides whether to call tools or write memory from inside its private
thinking block, and the orchestrator filters and executes those internal
directives before the visible answer is delivered.

`rule_planner.py`, `plan.py`, and `actions.py` remain as small compatibility
types around existing tool and memory action handlers.
