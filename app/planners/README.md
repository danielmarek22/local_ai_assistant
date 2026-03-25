
# Planners

Decision-making components.

## Responsibilities
- Analyzing current state and context
- Deciding next actions
- Selecting tools or LLM calls
- Requesting memory writes by emitting actions

Planners define the *behavior* of the assistant.
Multiple planners can coexist or be swapped.

## Current Planner Set

- `rule_planner.py`
  - heuristic routing for explicit memory commands and obvious web-search intents
- `llm_planner.py`
  - asks the model for a JSON action plan
- `hybrid_planner.py`
  - lets rules handle confident cases first, otherwise falls back to the LLM planner

Planners return `Plan` objects made of declarative actions such as:

- `web_search`
- `write_memory`
- `respond`
