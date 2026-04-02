# Tests Guide

This directory contains the baseline unit test suite for the Local AI Assistant.  
Tests currently focus on deterministic core behavior that can be validated without running the full server stack.

## What Is Covered

- `test_rule_planner.py`
  - Rule-based planner intent detection and action generation
  - Memory command extraction
  - Search intent routing and default response fallback

- `test_llm_planner.py`
  - LLM planner JSON parsing behavior
  - Handling extra text around JSON payloads
  - Fallback behavior when LLM output is invalid
  - Planner-specific non-streaming LLM option overrides

- `test_context_builder.py`
  - Prompt/context assembly order
  - Injected background context behavior
  - Summary inclusion behavior
  - Deduplication and filtering of recent history
  - Replaying recent stored image attachments into multimodal context

- `test_memory_store.py`
  - SQLite + vector-store dual writes
  - Semantic retrieval through the memory collection

- `test_chat_history.py`
  - Persisted attachment metadata and upload storage behavior
  - Image-summary writes into episodic memory
  - Session deletion cleanup across SQLite, uploaded files, and vector data

- `test_orchestrator.py`
  - Turn lifecycle and event emission
  - Memory/tool context injection into prompts
  - Avatar expression tag parsing
  - Image-only user turns and attachment persistence
  - Summarization trigger behavior

- `test_tool_executor.py`
  - Tool lookup and availability behavior
  - Error handling and fallback behavior

- `test_server_sessions.py`
  - Session listing/open/delete behavior
  - Session summary access
  - Restoring stored image attachments in session history
  - Local session switching semantics

- `test_http_retries.py`
  - LLM retry behavior
  - Web search retry behavior
  - Thinking-mode handling for Ollama

- `test_sentence_splitter.py`
  - Sentence chunking for streaming/TTS

## Running the Suite

Run all tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

Run a single test module:

```bash
python -m unittest tests.test_rule_planner -v
```

Run a single test case:

```bash
python -m unittest tests.test_rule_planner.RulePlannerTests.test_default_falls_back_to_respond_only -v
```

## Design Notes

- The suite uses Python's built-in `unittest` framework to keep dependencies minimal.
- Most tests use lightweight fakes/mocks to isolate behavior.
- Memory-related tests use in-memory SQLite plus fake vector-store collections for fast, repeatable runs.

## Adding New Tests

When adding tests, prefer:

1. Fast unit tests for deterministic logic (planner decisions, policies, formatting).
2. Integration-like tests with fakes for orchestrator turn flow.
3. Regression tests for discovered bugs before implementing fixes.

Suggested next additions:

- Websocket-level tests for streaming and TTS event sequencing.
- Browser-level tests for attachment and clipboard workflows.
- Tests around summarizer prompt structure and summary replacement behavior.
