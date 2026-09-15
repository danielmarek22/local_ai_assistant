---
title: Astra versus open-source agent architectures
description: Source-level comparison, reproduced gaps, and improvements that preserve custom orchestration.
---

# Astra versus open-source agent architectures

**Date:** 15 September 2026. **Decision:** retain custom orchestration; repair three narrow correctness gaps, then borrow execution contracts and lifecycle patterns selectively.

Future work is organized by milestone in the standalone [Future improvements document](future-improvements.md).

## Executive assessment

Astra already has useful separation between transport, turn coordination, inference, capabilities, storage, beliefs, and presentation. Replacing the backend would recreate many working behaviors inside another runtime while leaving Astra-specific authority and continuity semantics to implement again. The comparison does not establish a net benefit for that migration.

However, custom orchestration should not mean informal execution contracts. The most valuable lessons from the reviewed projects are explicit execution state, consistent tool metadata, cancellation ownership, structured observations, and budgets enforced across a complete run. These can be implemented within Astra's existing boundaries.

This investigation also changes the initial Milestone 0+ advice that there was nothing urgent to rework. Three isolated probes reproduced current gaps: tool allowlists are not enforced at dispatch, context reconstruction mixes message IDs and counts, and conflicting terminal operation results can overwrite each other. Fix those before adding more capable external tools. This does not invalidate Milestone 0's improvements; it narrows their previously overstated guarantees.

## Method, scope, and confidence

The local baseline is commit `5986652eca4d98fe778c8e51c37bd0515ffcf67a` **plus the working tree as inspected**, including pre-existing telemetry edits. No production code was changed by this review. File hashes are recorded in `reviews/milestone-0-plus/baseline.json`; probes and observed output are alongside it. References to local code below name files and symbols so they remain searchable as line numbers move.

Six projects were examined at selected implementation boundaries. This is a source-level architecture comparison, not an exhaustive audit of those projects and not a runtime benchmark. External references use moving `main` branches or documentation checked on this date; pin an upstream revision before copying code or running an adoption experiment. No external framework was installed, executed, or shown to outperform Astra. Direct shell network access was unavailable; official source files and documentation were inspected through web access.

Evidence labels:

- **Reproduced:** executed against Astra with in-memory databases or harmless fake handlers/models.
- **Source-confirmed:** a behavior or limitation visible in the inspected implementation, without an end-to-end reproduction.
- **Recommendation:** a proposed design or priority, not an assertion of existing behavior.

## Comparison matrix

| Project | Implementation examined | Architectural lesson for Astra | Adoption verdict |
| --- | --- | --- | --- |
| LangGraph | Retry/timeout implementation, execution types, persistence and interrupt contracts | Durable suspension, attempt identity, explicit retry boundaries | Borrow patterns; optional future Planner experiment |
| OpenHands SDK | Local conversation implementation and execution-server architecture | Execution state separated from agent decisions; pause versus interruption | Borrow task lifecycle and cancellation concepts |
| Pydantic AI | Tool definitions, function toolsets, usage accounting | Consistent typed tool/run contracts instead of special cases scattered through a loop | Borrow contracts; optional backend adapter experiment |
| smolagents | Agent loop and action-memory records | Small explicit steps with inspectable outcomes | Borrow step-record shape; keep Astra's tool-call model |
| Pipecat | Pipeline worker and frame lifecycle | End, cancel, interruption, and media health are different signals | Borrow presentation lifecycle; no immediate pipeline migration |
| Letta Code | Memory constraints and confinement entry point; current project documentation | Explicit memory budgets and separation between memory editing and authority | Borrow memory policy techniques; retain belief semantics |

LiveKit and Microsoft Agent Framework remain useful secondary references, but were not reviewed to the same implementation depth here. Adding them would not resolve the concrete gaps identified below faster.

## 1. Findings to address before expansion

### F1 — Enforce capability permissions at execution

**Implementation update:** fixed in the working tree. Invocation authority is frozen and enforced by the registry, including direct executor paths. Event calls missing explicit authority fail closed. Eight regression tests are in `tests/test_capability_authority.py`. Validation: 573 Python tests passed, critical lint and scoped type checks passed, and the strict documentation build passed. F2 is addressed below; F3 remains open. The original finding below records the reviewed baseline.

**Priority: first. Evidence: reproduced. Category: REFACTOR NOW within KEEP architecture.**

`IntegrationRegistry.get_native_tools()` filters schemas by `allowed_capabilities` and exposure checks. `ResponseGenerator._stream_late_routing_step()` accepts a returned registered capability without checking it against that permitted set. `_execute_late_tool_call()` passes it to `ToolExecutor.execute()`, and `IntegrationRegistry.invoke()` checks availability and schema but not the invocation's capability permission set. `InvocationContext` currently has no allowed-capability field.

The probe supplied an empty allowlist and a fake model returning `demo__blocked`. Both inference requests saw zero schemas, yet the harmless handler ran. Thus hiding a capability from the model is not an execution boundary. Event turns pass their declared allowlists into this same generation path.

This does **not** prove that shell approval or belief provenance can be bypassed: those handlers have additional checks. It proves that the general per-event capability restriction is not enforced centrally, so a model returning an omitted but registered tool can reach its handler.

**Change:** carry immutable application-owned capability authority to dispatch and reject unauthorized calls before handler invocation. Distinguish capability authorization from optional schema visibility; preserve separate handler-level belief and approval checks. Explicitly distinguish unrestricted trusted internal calls from an empty set, which must mean no tools.

**Acceptance:** a model requests a registered tool outside its event allowlist; its handler is never entered. Cover empty sets, allowed calls, disabled integrations, changed availability, and direct executor calls. Do not rely only on testing generated schemas.

Pydantic AI's prepared tool definitions and typed execution context are useful contract references, but do not substitute for defining Astra's own authorization policy. [Tool contracts source](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/tools.py).

### F2 — Complete the summary checkpoint migration in the reader

**Implementation update:** fixed in the working tree. The reader compares stored message IDs directly, using the same role/exclusion eligibility as summarization before applying the history limit. It retains the newest unsummarized messages within that bound and up to two recent continuity messages. Four real-store regression tests cover interleaved sessions, a high checkpoint, tool/system/excluded rows, bounded backlog, current summaries, and absent summaries. Validation: all 577 Python tests, critical lint, scoped type checks, and the strict documentation build passed. The original finding below records the reviewed baseline.

**Priority: first. Evidence: reproduced. Category: REPAIR.**

`SummaryStore.get()` returns `(summary, last_message_id)`. `TurnFinalizer.finalize()` correctly advances that ID using ordered summary batches. However, `ContextBuilder.build()` names the value `last_summarized_count`, and `_history_limit_for_summary()` subtracts it from `ChatHistoryStore.count_messages(session_id)`.

A message ID is global and can be much larger than the number of rows in a particular session. The probe inserted 100 messages in another session, summarized target message 101, then added four target messages. With a history limit of ten, context contained only the last two of those four messages. The first two were neither in the saved summary nor in recent context.

**Change:** select/count eligible messages after the checkpoint by ID; use the same eligibility policy as summarization. Rename the contract explicitly. Preserve the configured history bound and define what happens if unsummarized backlog exceeds it. Avoid assuming that subtracting a global ID from a session count has meaning.

**Acceptance:** integrate the real summary store, history store, and context builder in one test; cover interleaved sessions, excluded rows, tool rows, and a checkpoint far above the session's message count. The current checkpoint tests mainly exercise the writer and do not catch this reader mismatch.

This is a local consistency bug, not a reason to install a memory framework. It corrects the initial assessment's claim that restored-summary continuity was fully covered.

### F3 — Make operation state transitions explicit and monotonic

**Priority: next small fix. Evidence: reproduced. Category: REFACTOR NOW.**

`AutonomyStore.finish_operation()` prevents a late `pending` result from overwriting a terminal state and protects `cancelled`. Other statuses can overwrite each other. The probe wrote `success`, then `error`; the stored status became `error`. `begin_operation()` also uses `INSERT OR REPLACE`, so an accidental reused ID can reset an existing record.

This is a demonstrated store-contract weakness, not proof that a real external service has already delivered conflicting terminal results. Adapter-specific deduplication reduces some exposure, but should not be the only protection.

**Change:** define legal transitions such as `running → pending → success/error/denied/unavailable`, with cancellation rules and terminal conflict handling. Accept duplicate identical completions idempotently; journal conflicting late results without silently changing the accepted outcome. Deliberate corrections should be a separate operation or explicit reconciliation event. Preserve the existing early-completion versus late-pending protection.

**Acceptance:** duplicate completion, contradictory completion, completion after cancellation, repeated begin with the same ID, and an external result arriving before the initiating handler returns.

## 2. Durable execution: learn from LangGraph without replacing Astra

Astra's autonomy store persists event payloads and operation lineage. Its broker replays processing events only when their `ReplayPolicy` is safe. That is valuable crash recovery for an event queue. It does not persist the native loop's current messages, completed steps, approval decision, or remaining run budget. `ToolExecutor.execute()` creates a new invocation UUID each time; operation records omit the original argument payload and an enduring plan-step identity.

LangGraph's persistence model provides checkpoints for execution state; interrupts suspend work for later resumption. Resuming a node can repeat earlier code, so its documentation explicitly requires care with side effects before interrupts. Checkpointing alone does not make external actions execute exactly once. [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [interrupt semantics](https://docs.langchain.com/oss/python/langgraph/interrupts).

Its retry implementation separates attempt metadata and guards some writes after a timed attempt closes. This is useful inspiration for rejecting stale callbacks, not a guarantee that arbitrary external side effects are cancelled. [Retry implementation](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_retry.py).

**Recommendation for Astra:** before implementing durable plans, define a small persisted run/step model: task objective, run ID, step ID, attempt, status, action argument reference, approval reference, result reference, remaining budget, and recovery rule. Store only what resumption needs; do not start by serializing the entire orchestrator or every presentation event.

For an interrupted side effect, represent **outcome unknown** when neither local state nor the external system can establish completion. Query by a stable operation key where supported. Do not replay merely because a Python function failed to return.

**Keep** the custom orchestration policy. **Investigate** a LangGraph runner only if the planned task semantics require enough checkpoint machinery that maintaining it locally becomes more expensive. A graph execution thread ID would remain an execution identifier, not automatically Astra's Main/Work Thread product model.

## 3. Approvals and resource ownership: learn from OpenHands

`SessionTurnCoordinator` currently holds both session ownership and global turn capacity for the duration of a turn. In `app/server.py`, that scope includes generation and speech draining. The synchronous approval callback waits on an asynchronous browser decision. Consequently a waiting approval can retain the global turn slot even though the model is idle. User priority only applies before admission; it cannot preempt the current turn.

This is **source-confirmed behavior**, not the deadlock already repaired in Milestone 0. It limits responsiveness once substantial background work exists.

OpenHands separates conversation execution status from the agent's next action. Its local conversation code distinguishes pause at a step boundary from async interruption, passes a cancellation token, and repairs unmatched tool actions after interruption. The synchronous fallback still pauses cooperatively. Its confirmation flow is an architectural reference; Astra should preserve explicit approvals rather than copy any implicit-confirmation behavior. [Conversation implementation](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py).

**Recommendation:** distinguish session mutation ownership, model-call capacity, external-operation waiting, and media playback ownership. A task awaiting approval should eventually be durable `WAITING_APPROVAL` state rather than a blocked worker retaining model capacity. Releasing the whole session immediately would introduce races; first create a resumable step and a session revision check. Revalidate the approval's exact arguments and authority when execution resumes.

The current generator bridge uses `run_in_executor`. Cancelling the awaiting coroutine does not forcibly stop its synchronous worker. Introduce cooperative cancellation checks between generation/tool steps and backend-specific cancellation where possible. Do not label an operation stopped until its execution boundary confirms this or its result is safely quarantined.

OpenHands' agent-server architecture also separates remote/container execution from agent reasoning. Astra can borrow that boundary for shell operations while keeping its reasoning local. [Execution-server architecture](https://docs.openhands.dev/sdk/guides/agent-server/overview).

## 4. Tool contracts and observations: learn from Pydantic AI and smolagents

Astra's `ToolSpec` currently contains a capability ID, description, and input schema. `ToolResult` has useful statuses and diagnostics but primarily returns text. The native loop contains belief-specific correction logic, takes only the first returned tool call, and appends result text plus repeated evaluation instructions to the model conversation.

Pydantic AI's function toolset exposes declarative retry, preparation, approval, sequential-execution, and timeout settings. Its timeout path can abandon a thread and ask the model to retry—an example of why Astra must separately classify replay-safe actions. Borrow the contract organization, not blanket timeout retries. [Function toolset source](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/toolsets/function.py).

smolagents stores explicit action steps with timing, calls, errors, and observations, while its loop has step limits and interruption checks. These are useful small abstractions; its code-execution mode is not needed to obtain them. Its replay display is not a crash-resumption guarantee. [Action memory](https://github.com/huggingface/smolagents/blob/main/src/smolagents/memory.py), [agent loop](https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py).

**Recommended incremental changes:**

- Introduce typed `GenerationStep` and `RunOutcome` results instead of loose dictionaries and final text as the only completion signal. Distinguish completed, blocked, cancelled, failed, and budget-exhausted runs.
- Add a compact application-owned execution policy: deadline, retry classification, approval requirement, cancellation support, output budget, and concurrency class. Add fields when used, not a speculative plugin framework.
- Preserve a stable model-call/tool-call/operation relationship. For a model emitting multiple calls, explicitly reject the unsupported batch or serialize all calls with matching observations; silently selecting the first is a fragile cross-provider contract.
- Keep full results as artifacts when useful and inject bounded structured observations. An error code and retryability field should drive policy; prose should explain it.
- Replace broad `TypeError` compatibility fallbacks in schema discovery with an explicit adapter/protocol. A real implementation error should not silently select a less informative calling convention.

Belief-specific epistemology remains in the belief service. Generic execution should know that a tool has a bounded repair policy without needing to become the owner of belief rules.

## 5. Context and memory: retain Astra's semantics, strengthen budgets

Astra's canonical SQLite records and derived Chroma indexes are sound ownership choices. `MemoryStore` rehydrates retrieval candidates from SQLite; the belief tool validates authoritative evidence and checks whether an application already exists. Those are domain guarantees worth preserving. None of the inspected alternatives establishes a drop-in replacement for Astra's sender/subject/claim model.

Current Letta development is in Letta Code; older V1 memory documentation should not be mistaken for the current implementation. Its memory-constraints module enforces configurable file, core-memory, and depth budgets. Its confinement entry point delegates to a sandbox boundary and fails when the required backend is unavailable. These are useful examples of explicit policy around mutable memory. [Current project](https://github.com/letta-ai/letta-code), [memory constraints](https://github.com/letta-ai/letta-code/blob/main/src/memory-constraints.ts), [confinement entry point](https://github.com/letta-ai/letta-code/blob/main/src/memory-confinement.ts).

Astra currently bounds history by message count and integration context by characters. A count of six messages does not bound tokens if one message is huge. Tool observations are appended in full to the active loop; the 1,024-character truncation applies to persisted tool traces, not to the observation sent back to the model. `BashExecutionTool` captures stdout/stderr before returning, so downstream truncation alone would not bound process-output memory.

**Recommendation:** add a total prompt budget with reserved completion space and explicit allocation for identity, current input, recent history, beliefs, retrieved memory, schemas, and tool observations. Use model-specific token estimates where available; otherwise use conservative measurable limits and report the approximation. Cap command output while collecting it, not only after it returns. Keep larger content behind an artifact reference and a bounded reader.

Also make the trust boundary explicit. `ContextBuilder._build_system_message()` embeds retrieved memory and integration text in a system message and asks the model to use it; it does not contain the untrusted-context instruction claimed in the earlier assessment. Mark external/retrieved material as evidence rather than authority and enforce tools in code. Labels can help model behavior but are not an injection-proof boundary.

The first memory change is F2. More ambitious decay, reinforcement, or model-authored memory restructuring should wait for evaluation evidence.

## 6. Voice and embodiment: borrow Pipecat's lifecycle vocabulary

Astra already separates text completion from speech and bounds optional speech queues and drain time. Keep those improvements. `AudioDelivery` quarantines a timed-out engine because a native synthesis thread cannot be safely killed; switching frameworks does not remove that physical limitation.

Pipecat's worker distinguishes normal end, stop, cancellation, interruption, health monitoring, and stage metrics. It also differentiates a bounded cancel wait from normal end draining. These distinctions are more valuable to Astra than importing the whole voice runtime. [Worker implementation](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/pipeline/worker.py).

**Recommendation for M2:** attach an explicit utterance/turn generation ID to speech jobs and playback messages. Let cancellation invalidate that generation, flush pending speech, and suppress late audio/gestures without borrowing a later turn's identity. Once that is reliable, release model capacity before optional audio finishes while retaining media ownership independently.

Measure synthesis queue wait, synthesis duration, first playable audio, browser playback start, and interruption-to-silence. The in-progress telemetry already covers model and tool timing; extend it with media lifecycle measurements instead of adding a competing recorder. Process-isolated synthesis is a separate optional improvement if real hangs justify it.

## 7. External access: reusable components, Astra-owned policy

The current search summarizer explicitly requests omission of URLs and sources and passes only snippet content into its summarization prompt. This is an avoidable loss of provenance and an additional inference call before the main model reasons about results.

**Replace that interface first when M3 starts:** return bounded records containing URL, title, snippet, and retrieval time. Add explicit page fetching and reuse an extraction library. Summarize only when content size requires it, retaining source IDs. This can improve traceability and potentially reduce model work; latency gains must be measured rather than presumed.

MCP defines tool discovery, schema-based calls, and structured tool results. An adapter can translate those into Astra's existing capability/result boundary. It does not supply Astra's local permission policy. Apply F1 before exposing a larger catalog, and prefer one client adapter to hand-written service protocols. [MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).

For shell/browser integration, separately specify authority, workspace, process lifetime, output limits, and credentials. Current shell approval is not workspace isolation, and a timeout response is not evidence that every descendant process was terminated. Use a disposable execution environment when stronger containment is required. No such deployment or sandbox was implemented in this review.

## 8. One scenario, compared end to end

**Scenario:** gather information, request approval for a write, restart while waiting, resume, then cancel while an external action may still be running.

| Stage | Astra now | Improvement to borrow |
| --- | --- | --- |
| Accept input | Durable message and authoritative turn | Keep existing ownership |
| Select action | Registered schemas; F1 shows dispatch gap | Enforce invocation authority independently of model exposure |
| Run a step | Bounded in-memory native loop; operation journal | Typed step/result with enduring step ID |
| Await approval | Blocking callback inside admitted turn | Durable wait state with argument-bound approval |
| Restart | History restored; safe events replayed; Mindcraft has operation reconciliation | Explicit run checkpoint and backend reconciliation; do not replay unknown side effects |
| Resume | No general persisted native-loop continuation | Restore run state and budget, revalidate authority |
| Cancel | Async waiting and synchronous work have different lifetimes | Cooperative token, backend stop, terminal-state fence |
| Late completion | Pending-after-terminal protected; F3 exposes terminal conflict | Idempotent completion with explicit conflict handling |
| Explain outcome | Assistant text and traces | Structured outcome plus linked evidence/artifacts |

Mindcraft's existing `_reconcile_pending_operations()` already queries remote operation IDs and publishes recovered terminal results. Generalize that proven adapter pattern before inventing a second generic recovery mechanism.

## 9. Implementation order and decision gates

| Order | Work | Scope / expected effort | Success criterion |
| --- | --- | --- | --- |
| 1 | F1 dispatch authorization | Small-to-medium: contracts, dispatch, regression tests | Omitted unauthorized tool cannot execute |
| 2 | F2 checkpoint reader | Small: history query/reader contract and integrated tests | Every eligible unsummarized message within budget is restored |
| 3 | F3 terminal operation transitions | Small-to-medium: store transitions and adapter compatibility | Late/conflicting events cannot silently rewrite accepted outcomes |
| 4 | Typed step outcomes and output/prompt limits | Medium: generation boundary and tool observation handling | Clear stop reasons; oversized output remains bounded |
| 5 | Finish current telemetry and add focused evaluations | Small-to-medium beyond work already present | Comparable end-to-end measurements and failure fixtures |
| 6 | M1 profiles, M2 media ownership, M3 source-preserving research | Separate milestone-sized changes | Reuse current composition and presentation boundaries |
| 7 | Planner run-state prototype | Medium experiment before production design | Restart/approval/cancellation scenario passes without duplicate actions |

Effort is relative engineering scope, not a time estimate. Avoid mixing the first three fixes with a framework migration or broad module cleanup.

For the Planner experiment, compare a minimal custom runner and LangGraph only after writing the same acceptance fixtures. Keep models fake initially: side-effect counters, controllable completions, persisted approvals, and restart simulation test semantics cheaply. Then add representative Ollama tasks to compare first-response latency, total calls/tokens, correctness, and resource use.

Adopt a runner only if it removes meaningful local code while preserving authority, storage ownership, single-model scheduling, and observability. If both options still require most of Astra's lifecycle machinery, borrowing the design may be cheaper than adding the dependency.

## 10. Validation and limitations

The five focused suites executed were `test_context_builder`, `test_summary_checkpoints`, `test_integration_registry`, `test_autonomy_store`, and `test_autonomy_coordination`: **57 tests passed**. This is not a full regression run. The independent probes demonstrated behaviors those suites do not currently cover:

| Probe | Observed result |
| --- | --- |
| Empty allowlist, model requests registered harmless tool | Handler executed; schemas supplied to model were empty |
| Checkpoint 101, four later target messages, history limit ten | Only `new-2` and `new-3` restored; `new-0` and `new-1` omitted |
| Operation marked success, then conflicting error | Stored terminal status changed to error |

Reproduce from the repository root with `PYTHONPATH=. venv_app/bin/python reviews/milestone-0-plus/probes.py`. The script uses in-memory databases, a fake model, and a harmless handler; it does not contact services or execute commands. Its JSON is observational evidence, not a passing regression test. After fixes, convert each scenario into assertions of the intended contract.

No live model, speech engine, Minecraft server, browser automation, or external framework runtime was exercised. No comparative quality, speed, or reliability ranking is claimed. Upstream source inspection establishes useful patterns, not absence of bugs in those projects. Any copied implementation requires checking the chosen revision's license and dependencies; this report copies no upstream implementation.

**Final decision:** keep Astra's custom orchestration and domain services. Repair the demonstrated gaps immediately in follow-up work; improve contracts and lifecycle semantics incrementally. The strongest benefit from open source here is knowing which execution problems to design for, and reusing bounded components where they fit.
