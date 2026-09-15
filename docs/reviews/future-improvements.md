---
title: Future improvements inspired by open-source solutions
description: A milestone-aligned improvement backlog that preserves Astra's custom backend.
---

# Future improvements inspired by open-source solutions

**Date:** 15 September 2026. **Status:** proposed future work; no framework adoption or implementation is implied.

## Architectural decision

Astra will retain its custom orchestration layer. Its identity, belief provenance, continuity, context-selection rules, permission decisions, and embodiment semantics remain application-owned. Existing open-source solutions provide components to reuse and architectural patterns to learn from.

This document collects future improvements from the September 2026 code comparison. It can be used independently as a planning reference: each item identifies the problem, proposed change, relevant external reference, milestone, and completion criteria. Technical observations describe the reviewed baseline (`5986652` plus the then-current working tree), not a claim that every limitation remains present after later development.

The three immediate findings—execution-time capability authorization, summary-checkpoint reading, and terminal operation transitions—are tracked separately. They are prerequisites where relevant, not part of this future backlog. Their current completion status should be checked before implementation begins.

## How to use this backlog

There are three different kinds of improvement:

| Approach | Meaning |
| --- | --- |
| **Reuse a component** | Adopt a library or protocol implementation behind Astra's interfaces. |
| **Borrow a pattern** | Improve Astra's implementation using a design demonstrated elsewhere. |
| **Compare before adopting** | Run a bounded experiment because dependency and migration costs may outweigh benefits. |

Implementation belongs to the milestone that needs the behavior. A small prerequisite can be implemented earlier when required by active work. This document is not an instruction to implement every item during Milestone 0+.

## Milestone overview

| ID | Improvement | Approach | Primary milestone |
| --- | --- | --- | --- |
| FI-01 | Explicit runtime profiles | Borrow a pattern | M1 — Runtime profiles |
| FI-02 | Independent media ownership and interruption | Borrow a pattern | M2 — Embodiment |
| FI-03 | Structured search and page reading | Reuse components | M3 — Information access |
| FI-04 | Prompt and tool-output budgets | Borrow a pattern | M3; extend in M4 |
| FI-05 | Typed steps, observations, and outcomes | Borrow a pattern | M4 — Planner |
| FI-06 | Consistent tool execution policy | Borrow a pattern | M4–5 — Planner / Safe agency |
| FI-07 | Durable tasks and approval waits | Compare before adopting | Before/during M4 |
| FI-08 | Cancellation and resource ownership | Borrow a pattern | M4–5; media portion in M2 |
| FI-09 | Isolated execution environments | Reuse components; borrow boundaries | M5 — Safe agency |
| FI-10 | Standard external tool connections | Reuse protocol implementations | M8, or earlier for a concrete integration |
| FI-11 | Scoped continuity and memory policy | Borrow patterns; retain domain design | M6 / M10 |
| FI-12 | Comparable measurements and evaluations | Extend existing infrastructure | Alongside each relevant milestone |

## FI-01 — Explicit runtime profiles

**Problem:** Personal, Demo, and Relay operation need different state, capabilities, and presentation. Distributing mode checks throughout the runtime would make ownership harder to reason about.

**Proposed change:** compose each profile at startup using the existing application factory and lifecycle boundaries. Select stores, prompt, enabled capabilities, and presentation explicitly. A Relay profile should activate local cognition only when deliberately delegated; a Demo profile should use disposable state.

**Reference:** Pydantic AI's typed context and dependency-oriented interfaces offer a useful separation pattern. The three Astra profiles are an Astra product decision, not a framework feature to copy. [Tool/context contracts](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/tools.py).

**Done when:** profile selection reliably determines state and capabilities; Demo cannot access personal records; Relay preserves external source identity; all profiles share the same relevant tested services. No new framework is required.

## FI-02 — Independent media ownership and interruption

**Problem:** Text completion is already decoupled from speech synthesis, but turn ownership still spans bounded speech draining. Audio, model work, and interruption need distinct lifecycles.

**Proposed change:** associate speech jobs, playback messages, and gestures with an explicit turn/utterance generation ID. Cancellation invalidates that generation, flushes queued output, and suppresses late results. Release model capacity independently once media attribution is reliable.

**Reference:** Pipecat distinguishes normal completion, cancellation, interruption, and processor health. Borrow those distinctions while retaining Astra's speech engines and renderer. [Pipeline worker](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/pipeline/worker.py).

**Done when:** interruption stops applicable playback, late audio cannot appear in another turn, and optional speech does not retain model capacity. Text-only operation must continue to work.

**Optional follow-up:** isolate synthesis in a killable worker process if actual engine hangs justify it. Neither a new framework nor async cancellation can forcibly stop arbitrary native code in an existing thread.

## FI-03 — Structured search results and explicit page reading

**Problem:** the reviewed search path compresses snippets through an extra model call and explicitly omits source URLs. Astra cannot inspect the underlying page through that interface.

**Proposed change:** retain SearXNG and return bounded records with source ID, URL, title, snippet, and retrieval time. Add a separate page-reading capability. Reuse an extractor such as Trafilatura; summarize only when needed, retaining source references. Add browser automation through Playwright later if JavaScript interaction or actual page actions become necessary.

**References:** [Trafilatura extraction](https://trafilatura.readthedocs.io/en/latest/), [Playwright browser contexts](https://playwright.dev/python/docs/api/class-browsercontext). These were identified as reuse candidates; their suitability has not been benchmarked in Astra.

**Done when:** an answer can be traced to retrieved sources, fetch failures are explicit, long pages remain bounded, and a representative research task can inspect full content. Measure any latency improvement rather than assuming that removing a summarization call always improves the result.

## FI-04 — Prompt and tool-output budgets

**Problem:** message counts and individual character limits do not enforce a total prompt budget. A single large message or tool observation can dominate context; capturing unlimited command output consumes memory before later truncation can help.

**Proposed change:** allocate a total budget across identity, current input, history, beliefs, retrieval, schemas, and observations, reserving completion capacity. Use model-aware estimates where possible and document conservative approximations otherwise. Bound tool output during collection; store useful larger results as artifacts with a bounded reader.

**References:** Letta Code makes memory limits explicit; Pydantic AI separates run usage from usage limits. These are complementary ideas: memory size and total execution consumption are different budgets. [Memory constraints](https://github.com/letta-ai/letta-code/blob/main/src/memory-constraints.ts), [usage accounting](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/usage.py).

**Done when:** oversized input/results produce predictable truncation or an explicit refusal; mandatory context and completion space are accounted for; dropped content is observable. Retrieved text must be treated as evidence, with executable authority enforced separately in code.

## FI-05 — Typed execution steps, observations, and outcomes

**Problem:** the native loop uses loose result dictionaries and final text as its main completion signal. This makes future task state and recovery difficult to inspect consistently.

**Proposed change:** introduce small typed records for a generation step, tool observation, and run outcome. Carry call/operation identity, status, diagnostics, timing, and artifact references. Represent completed, blocked, failed, cancelled, and budget-exhausted outcomes explicitly. Preserve matching observations for every accepted tool call.

**Reference:** smolagents records explicit action steps with calls, observations, errors, and timing. This pattern does not require adopting its code-execution mode. [Action-step records](https://github.com/huggingface/smolagents/blob/main/src/smolagents/memory.py).

**Done when:** the UI and future Planner can determine the actual run outcome without interpreting prose. Multiple model-emitted calls receive an explicit policy—rejection or supported serialization—rather than silently losing calls after the first.

## FI-06 — Consistent tool execution policy

**Problem:** deadlines, correction rules, approval requirements, and other execution behavior are spread across the runtime and particular tools.

**Proposed change:** add application-owned policy metadata as needed: deadline, retry classification, approval requirement, output limit, concurrency class, and cancellation support. Keep domain validation inside its service; use typed diagnostics to distinguish a correctable argument error from a terminal execution failure. Replace broad compatibility fallbacks with explicit interfaces as affected code is changed.

**Reference:** Pydantic AI's function toolsets collect preparation, retry, approval, sequencing, and timeout settings at a defined boundary. [Function toolsets](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/toolsets/function.py).

**Done when:** equivalent failures follow consistent policy across entry paths. Timeouts do not automatically authorize replay of side effects, and model-visible schemas do not determine permissions. This extends the immediate authorization repair; it should not delay that repair.

## FI-07 — Durable tasks and approval waits

**Problem:** a durable event journal is not a persisted task continuation. The reviewed native loop does not save its steps, approval decision, remaining budget, and resume point as a general task model.

**Proposed change:** define a minimal persisted run/step contract. Include objective, run/step/attempt IDs, status, action and result references, approval binding, budget, and recovery rule. A task awaiting approval should become a durable waiting state instead of retaining a blocked worker and model capacity.

**References:** LangGraph supplies persistence and resumable interrupts. Resumed code can repeat work before the interruption, so checkpointing does not provide exactly-once external effects. OpenHands offers useful execution-state separation. [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [OpenHands conversation](https://docs.openhands.dev/sdk/arch/conversation).

**Decision gate:** compare a small custom runner and a LangGraph prototype using the same task: tool call, approval wait, restart, resume, cancellation, late completion. Adopt a runner only if it removes substantial maintenance without taking over Astra's domain semantics or creating a second competing scheduler.

**Done when:** the task resumes with preserved authority and budget, approvals apply to the exact proposed action, and uncertain external effects are reconciled or marked outcome-unknown rather than blindly replayed.

## FI-08 — Cancellation and resource ownership

**Problem:** session serialization, model capacity, external waiting, and media ownership currently overlap. Cancelling an async wait does not necessarily stop the underlying synchronous work.

**Proposed change:** distinguish those resource lifetimes. Add cooperative cancellation between steps and backend-specific stopping where supported. Use attempt/generation identity to reject stale callbacks. A waiting task may release model capacity while retaining a durable continuation and protecting session consistency.

**References:** OpenHands distinguishes pause from interruption and propagates cancellation to execution; LangGraph's retry implementation demonstrates attempt-scoped handling. [OpenHands local conversation](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py), [LangGraph retry implementation](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_retry.py).

**Done when:** waiting work does not unnecessarily block unrelated model calls; cancellation prevents later steps; late results cannot mutate superseded work. Clearly distinguish requested cancellation from confirmed stopping. Preserve terminal-state protection established by the immediate fix.

## FI-09 — Isolated execution environments

**Problem:** approval establishes authorization, not filesystem or process isolation. Longer autonomous tasks need explicit ownership of working directories, subprocesses, outputs, and credentials.

**Proposed change:** introduce an execution backend boundary. Use a suitable existing container/sandbox mechanism when containment is needed, with explicit workspace access, process-tree termination, output limits, and cleanup. Keep Astra's approval decisions outside that backend.

**References:** OpenHands separates execution servers from agent reasoning; Letta Code's memory-confinement entry point illustrates failing when a required isolation boundary is unavailable. [Execution server](https://docs.openhands.dev/sdk/guides/agent-server/overview), [memory confinement](https://github.com/letta-ai/letta-code/blob/main/src/memory-confinement.ts).

**Done when:** approved work stays within the configured execution boundary, cancellation cleans up owned processes, and artifacts remain attributable. Choose and validate a concrete backend during M5; this document does not select one.

## FI-10 — Standard external tool connections

**Problem:** hand-writing every external-service integration duplicates discovery, transport, schema handling, and connection lifecycle.

**Proposed change:** add an MCP client adapter that maps supported external tools into Astra's registry and structured results. Retain local capability authorization, approval, deadlines, output bounds, and source identity. Define behavior for changed schemas, unavailable servers, and reconnects.

**Reference:** [MCP tool specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools). Reuse a maintained implementation after checking compatibility; the protocol alone is not a permission engine.

**Done when:** a real service can be connected through the adapter with discovery, success, failure, timeout, approval, and disconnect scenarios covered. Implement when there is a concrete service to connect, rather than building an abstract integration ecosystem first.

**Later investigation:** A2A may be appropriate for communication with an actual compatible agent. It is a separate task/agent protocol question and does not replace MCP or establish the truth of remote claims. [A2A concepts](https://a2a-protocol.org/latest/topics/key-concepts/).

## FI-11 — Scoped continuity and deliberate memory policy

**Problem:** future work threads need task-local detail without contaminating the Main Thread. Durable identity and beliefs must retain attribution across those contexts.

**Proposed change:** preserve canonical storage and source-aware beliefs. Introduce structured work-thread checkpoints containing objective, outcome, decisions, artifacts, and unresolved items. Make durable knowledge promotion deliberate. Apply explicit memory limits and evaluate retrieval behavior before adding decay or reinforcement mechanisms.

**Reference:** Letta is useful for studying persistent-agent memory, but Astra's Main/Work Thread model and epistemic rules remain its own design. [Letta Code](https://github.com/letta-ai/letta-code).

**Done when:** a completed Minecraft or debugging task returns useful conclusions without injecting its entire transcript into Main context; source attribution and visibility survive promotion. A display-name change should eventually preserve participant identity under M10's identity model.

## FI-12 — Measurements and evaluation fixtures

**Problem:** structural similarity to another framework does not prove better behavior. Performance and correctness need repeatable measurements.

**Proposed change:** extend the telemetry work already present in the reviewed workspace. Add queue wait, first visible text, first playable audio, interruption latency, per-step outcomes, and resource usage as the corresponding features arrive. Build deterministic fixtures for restart, approval, cancellation, duplicate results, and stale observations, then representative live-model evaluations.

**References:** Pydantic AI's usage accounting and Pipecat's lifecycle metrics offer useful distinctions. [Usage model](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/usage.py), [media worker](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/pipeline/worker.py).

**Done when:** a proposed dependency or refactor can be compared against the same baseline scenarios. Record correctness alongside latency and token counts. Avoid a second telemetry stack when the existing recorder can carry the required measurements.

## Scheduling and adoption rules

Finish the three immediate repairs separately. Then implement M1 profiles, M2 embodiment ownership, and M3 information access in their normal sequence. Design the M4 execution contract using the later items above; run the durable-runner comparison before committing to its implementation.

Reuse a component when it fits a narrow boundary and removes measurable maintenance. Borrow a pattern when Astra's semantics remain specific but another project offers a clearer implementation model. Adopt a framework only after a prototype demonstrates value beyond a feature checklist.

Manifest-backed asset catalogs remain an M2 improvement from the original roadmap. They are a local product/interface decision rather than a finding demonstrated by these agent-framework comparisons. Broader memory decay, self-modification, and always-on Executive behavior remain research questions until a concrete need and evaluation justify them.

## Evidence and companion documents

This is a planning synthesis of the [source-level comparison](framework-comparison.md) and [Milestone 0+ inventory](../milestone-0-plus.md). It does not report new experiments or change the status of existing fixes. External references reflect the earlier review and may move; verify versions, licenses, and dependencies when an implementation starts. No benchmark has established that any listed framework is faster or more reliable than Astra.
