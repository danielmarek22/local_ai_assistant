
# Services

Support utilities and long-lived helper services used by the orchestrator.

## Responsibilities
- Prompt/context assembly
- Search result summarization
- History summarization
- Image summarization for long-term retrieval
- Memory retrieval helpers
- Turn-finalization helpers
- Tool execution helpers
- Text streaming helpers such as sentence splitting

## Current Notes

- `context_builder.py` builds the final prompt from:
  - the system prompt
  - current local datetime
  - configured user context
  - injected background context (retrieved memory + observed integration state)
  - session summary
  - recent chat history
- `memory_retriever.py` combines semantic memory and episodic history retrieval into one reusable turn helper.
- `memory_action_handler.py` applies structured memory writes from the `memory__write` integration.
- `turn_finalizer.py` owns rolling-summary updates after a turn completes.
- `stream_processor.py` lives under `app/core` because it is turn-stream parsing logic, but it serves the same “small helper around orchestration” role.
- `image_summarizer.py` generates one concise factual summary per stored image so attachments can be found again through episodic vector search without resending every old image to the model.
- `tool_executor.py` adds assistant events and tracing around validated integration calls.

Some files in this directory are classic "services", while others are prompt and
execution helpers. In practice, this folder currently groups assistant support
logic that does not fit cleanly into `core`, `memory`, `tools`, or `llm`.
