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

- `test_context_builder.py`
  - Prompt/context assembly order
  - Tool context injection
  - Memory and summary inclusion behavior
  - Deduplication and filtering of recent history

- `test_memory_store.py`
  - Relevance ranking logic for memory retrieval
  - Importance handling when lexical overlap is low

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
- `MemoryStore` tests use an in-memory SQLite database (`:memory:`) for fast, repeatable runs.

## Adding New Tests

When adding tests, prefer:

1. Fast unit tests for deterministic logic (planner decisions, policies, formatting).
2. Integration-like tests with fakes for orchestrator turn flow.
3. Regression tests for discovered bugs before implementing fixes.

Suggested next additions:

- `Orchestrator` turn lifecycle tests (state events, action execution order).
- `ToolExecutor` behavior tests (unavailable tools, failures, fallback paths).
- `sentence_splitter` edge-case tests.
- Websocket-level tests for streaming and TTS event sequencing.
