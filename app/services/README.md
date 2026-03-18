
# Services

Support utilities and long-lived helper services used by the orchestrator.

## Responsibilities
- Prompt/context assembly
- Search result summarization
- History summarization
- Tool execution helpers
- Text streaming helpers such as sentence splitting

## Current Notes

- `context_builder.py` builds the final prompt from:
  - the system prompt
  - current local datetime
  - configured user context
  - injected background context (retrieved memory + tool results)
  - session summary
  - recent chat history
- `tool_executor.py` is the adapter that turns planner actions into tool calls.

Some files in this directory are classic "services", while others are prompt and
execution helpers. In practice, this folder currently groups assistant support
logic that does not fit cleanly into `core`, `memory`, `tools`, or `llm`.
