---
title: Milestone 0+ — Architectural sanity check
description: What Astra should keep, refactor later, replace, or investigate before further expansion.
---

# Milestone 0+ — Architectural sanity check

**Assessment date:** 14 September 2026. **Code baseline:** `5986652`. **Status:** analysis complete; implementation decisions below are recommendations, not completed migrations.

Astra should keep its domain-specific core and reuse external infrastructure at explicit boundaries. The current evidence does **not** justify replacing the whole runtime with an agent framework. The strongest reuse opportunities are standard external tool connections, webpage extraction, and eventually browser automation. Durable planning deserves a comparative prototype before its implementation is committed to a framework or built from scratch.

!!! warning "Follow-up findings — 15 September 2026"
    The [source-level framework comparison](reviews/framework-comparison.md) reproduces gaps in dispatch allowlist enforcement, summary-checkpoint reading, and terminal operation transitions. Its findings qualify the initial assessment below and take priority over broad claims that these guarantees were fully solved. The custom backend remains the recommended foundation.

## Purpose and evidence

In **“Roast Of AI Project”**, the user clarified that Milestone 0+ means checking feasibility and existing implementations before building more infrastructure. The subsequent discussion framed it as deciding what Astra should own versus reuse. This supersedes the earlier interpretation of 0+ as an external-interaction implementation milestone.

This assessment uses that discussion, **“Project Astra Roadmap”**, available outputs from **“Scan Astras roadmap codebase”**, the current source, regression-test inventory, commit history, and official ecosystem documentation checked on the assessment date. **“Review Small Astra architecture”** was located, but its recent turns returned no message content; no conclusions are attributed to those inaccessible messages. The historical `astra-stabilization-audit.md` was not available among the repository's tracked documents, so committed fixes are the stabilization evidence here.

Milestone 0 is accepted as complete per the user's instruction. “Solved” below means a concrete implementation/fix is present with relevant regression coverage; it does not mean every model, device, or external service was retested during this analysis. No live-model benchmark or framework migration was performed.

The categories apply to specific layers. A feature can have a retained domain contract and a replaceable implementation underneath it:

| Category | Decision rule |
| --- | --- |
| **KEEP** | Preserve the current capability or invariant; custom ownership has a clear purpose. |
| **REFACTOR LATER** | Preserve behavior; improve structure when a named milestone needs it. |
| **REPLACE** | Retire a specific current approach in favor of the identified direction, with a migration gate. |
| **INVESTIGATE** | Evidence is insufficient to choose; run a bounded comparison before committing. |

## What Milestone 0 has already solved

These are foundations to carry forward, not tasks to rebuild under a new name.

| Previously problematic area | Present solution and evidence | Remaining boundary |
| --- | --- | --- |
| Restored/long conversations losing continuity | Bounded context, rolling summaries that include the previous summary, and message-ID checkpoints. Commits `6a27872`, `a3c1ac0`; `tests/test_context_builder.py`, `tests/test_summary_checkpoints.py`. | Summary quality and relevance still depend on the model. This does not provide Main/Work Thread semantics. |
| Old images and enrichment delaying turns | Historical image context is bounded; image summarization is deferred and bounded. Commit `8c8f746`; context and image-fallback tests. | Vision quality and GPU capacity are separate concerns. |
| Retrieval failure losing an accepted input | Input becomes durable before optional retrieval; retrieval sources fail independently. Commit `60bf603`; `tests/test_retrieval_failures.py`. | A failed index can reduce context quality even when conversation continues. |
| Stale vector results acting as truth | Retrieval candidates are checked against canonical SQLite records; index reconciliation exists. Commit `785fc67`; `tests/test_vector_candidates.py`. | Index maintenance and relevance evaluation remain useful. |
| Memory consolidation inconsistency | Reflection response handling and atomic canonical consolidation were repaired. Commit `1315293`; `tests/test_memory_reflector.py`. | Useful merging is not the same as a validated decay/reinforcement policy. |
| Reconnects losing turn/input state | Active turns and input acknowledgement survive reconnects. Commits `bcf7cd7`, `fe43660`; server, network-client, and acknowledgement tests. | This is not durable arbitrary task execution across process crashes. |
| Deleted sessions being recreated | Session lifecycle guards prevent resurrection by late work. Commit `2988da9`; `tests/test_session_deletion.py`. | Future thread migration must preserve these guards. |
| User/event scheduling deadlock | Shared admission checks avoid holding a session while waiting for model capacity. Commit `0712cb7`; `tests/test_autonomy_coordination.py`. | User priority applies at turn boundaries; active inference is not preempted. |
| Speech blocking text completion | Response generation and audio ownership are separated; synthesis and drain waits are bounded. Commits `490e384`, `f6d7221`; `tests/test_audio_delivery.py`, `tests/test_speech_stream.py`. | A stuck native engine cannot be killed in-process; restart is needed after its deadline disables speech. |
| Fragile contracts and lifecycle | Strict configuration/WebSocket validation, application-owned resources, serialized SQLite transactions, foreign keys, package-boundary tests, and CI lint/type checks exist. | These are engineering controls, not proof of all end-to-end behavior. |
| Belief rejection and misleading repair guidance | Grounding, memory guidance, invalidation contract, and repair feedback have recent fixes (`050225f`, `5986652`); belief/react tests cover deterministic rules. | Model willingness and extraction accuracy still need realistic evaluation. |
| Repeated one-shot gestures not replaying | Gesture replay was repaired in `672caba`; browser gesture tests exist. | Coordinated embodiment and a persistent 3D environment remain later work. |

## KEEP

| Feature / layer | Why Astra should own or retain it | Reuse boundary / evidence |
| --- | --- | --- |
| **Belief semantics and provenance** | Exact source evidence, sender/subject distinction, coexisting attributed claims, visibility, and invalidation are central Astra requirements. Generic persistence does not supply these semantics. | Keep `app/beliefs/` and authoritative mutation scope. A framework may call the service; it must not manufacture provenance. |
| **Canonical SQLite history and memory** | Already separates durable records from fallible retrieval indexes and has deletion/reconciliation behavior. | Keep `app/storage/`, `app/memory/`; Chroma remains a replaceable derived index. |
| **Context reconstruction and budgets** | Stateless inference needs explicit continuity, bounded images/history, and separation of untrusted retrieved context. These policies remain necessary with any model SDK. | Keep `app/core/context_builder.py`; adapt future thread summaries into it. |
| **Authoritative turn and participant input** | Application-owned session, message, sender, and time are essential to safe belief/tool attribution. | Keep `TurnInput` and `AuthoritativeTurnContext`; transport adapters normalize into them. |
| **Validated internal capability registry** | Schema validation, availability, typed results, effects, and per-event allowlists are already useful. | Keep `app/integrations/registry.py` and contracts; expose external standards through adapters. |
| **Bounded native tool loop and instant mode** | Already supports tool/result iteration and direct low-latency conversation. No need to introduce a framework just to regain these behaviors. | Preserve behavior in `ResponseGenerator`; this is not yet a durable high-level Planner. |
| **Durable event journal and model coordinator** | Pending operations, correlated completion events, bounded event turns, and notification policy already exist. | Keep `app/autonomy/` as the current baseline; do not build a second scheduler before comparing ownership. |
| **Speech and avatar presentation** | Voice, expressions, FBX gestures, outfit changes, and text-first completion are distinctive working features. | Keep existing STT/TTS engines and Three.js/VRM rendering behind presentation contracts. |
| **Mindcraft adapter and typed control** | Existing direct operations, pending results, events, handshake, and external/hybrid modes already absorb substantial integration work. | Keep `app/integrations/mindcraft.py`; reassess the backend only after Planner trials. |
| **Knowledge inspector and diagnostics** | Inspectable beliefs/memories and traces make behavior understandable and future migrations reviewable. | Preserve read-side knowledge APIs and their privacy/ownership tests. |
| **Documentation and regression infrastructure** | The source-controlled architecture map and deterministic checks protect precisely the fixes a migration might lose. | Keep MkDocs, dependency manifests, Python/browser tests, and CI gates. |

## REFACTOR LATER

| Feature / layer | Change when needed | Trigger and acceptance condition |
| --- | --- | --- |
| **Runtime assembly and profiles** | Use explicit Personal/Demo/Relay composition to select prompts, stores, capabilities, and presentation. Build on the existing factory/lifespan ownership. | **M1:** demo uses disposable state; relay does not accidentally activate local cognition. Avoid scattered mode checks. |
| **Conversation routing** | Introduce Main/Work Thread concepts above storage and normalized input, retaining message IDs and provenance. | **M6:** work results return as structured checkpoints; full work transcripts do not leak into Main context. |
| **Tool execution policy** | Centralize risk/workspace/approval policy as capabilities expand; retain the conservative shell approval behavior. | **M4–5:** enforcement covers every entry path, including headless/event calls. Approval is not an OS sandbox. |
| **Embodiment coordination** | Unify timing, semantic expressions, gestures, gaze, interruption, and optional scene ownership. | **M2:** text-only operation stays functional and late audio never acquires another turn's identity. |
| **Perception contracts** | Evolve existing attachments/frame controller/watchdog into timestamped source-aware observations with demand-driven interpretation. | **M9:** freshness and origin survive context injection; unchanged inputs avoid unnecessary inference. |
| **Integration registration** | Add discovery only when external integrations justify it; explicit built-in registration is currently understandable. | **M8:** validated lifecycle and capability contracts remain mandatory for discovered adapters. |
| **Memory selection/consolidation policy** | Tune retrieval and merging using observed errors; keep canonical storage and belief/memory separation. | Repeated relevance failures with an evaluation set, rather than speculative memory-decay machinery. |

## REPLACE

These are bounded replacement directions, not a mandate to remove whole subsystems now.

| Current approach | Replacement direction | What survives / migration gate |
| --- | --- | --- |
| **Search snippets compressed into one answer as the research interface** | Structured search results plus an explicit bounded page reader. Reuse an HTML extraction library such as **Trafilatura** instead of writing a general extractor. | Keep SearXNG and the capability registry. Preserve URL/title/retrieval time, readable text, limits, and explicit fetch failure. Compare extraction on representative pages before adoption. Trafilatura already exposes text/metadata extraction. [Official docs](https://trafilatura.readthedocs.io/en/latest/). |
| **Filename-derived outfit/gesture identity as the long-term asset contract** | Manifest-backed semantic catalogs, with stable IDs, enabled state, display metadata, and validated asset references. | Keep outfit tool and renderer. Current discovery already exists in `app/avatar/controls.py`; it scans VRM/FBX names, so discovery itself is not missing. Migrate existing IDs without breaking calls; persist appearance deliberately. |
| **Display-name-derived participant identity for future social use** | Stable participant IDs with display names as mutable labels and explicit transport mappings. | Keep source-aware beliefs and manual group use. M10 migration must preserve old attribution; a rename must not silently become a different participant. |
| **Independent equal-status sessions as the final continuity model** | One Main Thread plus scoped Work Threads and structured result summaries, as already requested in the roadmap. | Reuse history, IDs, summaries, and deletion guards. M6 requires migration rules and task-local knowledge policy; do not flatten existing history into a new transcript. |

The last three replacements are Astra product/domain decisions; an external framework will not choose their semantics for us. No whole-runtime replacement has enough evidence for an unconditional **REPLACE** verdict today.

## INVESTIGATE

| Question / feature | Existing solution worth comparing | Decision experiment and provisional verdict |
| --- | --- | --- |
| **Durable Planner and task execution** | **LangGraph** provides checkpointed graph execution and interrupts; **Pydantic AI** offers Ollama integration and durable-execution integrations. [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [Pydantic Ollama](https://pydantic.dev/docs/ai/models/ollama/), [durability](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/). | Before M4, implement the same small plan in an isolated prototype: tool → approval wait → restart → resume → cancellation. Require preserved authority, bounded local-model use, correct outcomes, and no duplicated side effects. Compare with extending current contracts. **Do not select a framework on feature-list overlap alone.** |
| **Long-running scheduling and recovery** | **Temporal** supplies a durable workflow engine and Python SDK. [Official SDK](https://github.com/temporalio/sdk-python). | Compare only when tasks must survive long waits/restarts beyond the present event queue. Measure operational/dependency cost and decide which component owns scheduling and recovery. **Defer adoption** for the current single-machine event runtime. |
| **External tool/service connectivity** | **MCP** defines host/client/server boundaries and tool access; transport authorization is standardized separately. [Architecture](https://modelcontextprotocol.io/specification/2025-11-25/architecture), [authorization](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization). | Preferred reuse direction: one MCP client adapter into Astra's registry. Trial discovery, call, error, timeout, and approval with a controlled server. Retain Astra's local authority, visibility, and budgets. MCP does not replace beliefs or a Planner. |
| **External agent communication** | **A2A** defines remote agent discovery metadata, tasks, artifacts, and asynchronous updates. [Core concepts](https://a2a-protocol.org/latest/topics/key-concepts/). | Test against one actual compatible peer before implementation. Preserve remote identity, source attribution, task ownership, and terminal results. **Investigate**, not a new bespoke agent-network protocol. A2A is not proof that a remote claim is true. |
| **Browser interaction** | **Playwright** supplies browser/page automation and separate browser contexts. [BrowserContext](https://playwright.dev/python/docs/api/class-browsercontext). | When needed, wrap Playwright or a compatible existing tool server. Trial session isolation, cancellation, download ownership, and approval for actions. Browser contexts separate browser state; they are not a filesystem/process sandbox. Page reading should come first. |
| **Mindcraft versus direct Mineflayer / specialist model** | Current fork already supports external control and hybrid delegation. | Run the same bounded Minecraft goal in both existing modes; compare latency, completion, recovery, and stop behavior. Direct Mineflayer is justified only if concrete limits outweigh lost integration plumbing. No rewrite now. |
| **Belief producer and model reliability** | Existing observer and react-tool modes provide two comparison paths. | Use examples of corrections, jokes, third-party claims, invalidations, and memory-worthy events; measure accepted correctness, missed updates, rejection/recovery loops, and latency. Deterministic fixes do not prove model behavior is solved. |
| **Model/backend and hardware fit** | Existing Ollama adapter is the baseline; alternative adapters can sit behind it. | Measure realistic-context time to first text, complete-turn time, tool accuracy, RAM/VRAM, and speech contention. Do not change inference backend or buy architectural complexity without measured benefit. |
| **Hard recovery of stuck speech engines** | Process isolation can provide a killable worker boundary. | Only pursue if restart-required failures are frequent enough to justify extra IPC/lifecycle complexity. Current bounded waiting already protects text completion. |
| **Memory frameworks, decay, fine-tuning, curiosity, Executive behavior** | These address different needs from proven storage and event primitives. | Keep as research questions until an evaluation demonstrates a specific missing behavior. Generic “agent memory” is not a substitute for Astra's belief provenance. |

## What Astra should own versus reuse

**Own:** identity and participant semantics; belief epistemology/provenance; canonical continuity; context selection; task objectives and outcomes; permission decisions; notification policy; embodiment meaning.

**Reuse:** database/vector engines, model and speech backends, rendering, content extraction, browser automation, external tool/agent wire protocols, and—if a prototype earns it—durable execution infrastructure.

The external ecosystem solves many mechanics, but those mechanics do not settle Astra's product semantics. Conversely, having custom semantics does not justify writing a browser driver, HTML extractor, or external interoperability protocol from scratch.

## Recommended next sequence

1. Carry this baseline into **M1 runtime profiles**; reuse the current lifecycle and presentation boundaries.
2. At **M2**, retain working avatar behavior and replace filename semantics with manifests as needed.
3. At **M3**, replace snippet-only research with structured results and library-backed page reading.
4. Before **M4 implementation**, run the durable Planner comparison. Choose one execution owner; avoid overlapping framework and Astra schedulers.
5. Add MCP, A2A, browser, and broader integration support when a concrete connection requires them, using the investigation gates above.

**Milestone 0+ exit result:** the inventory, reuse directions, replacement targets, and unresolved decision gates are recorded. No framework migration is presumed approved or implemented. Milestone 0's reliability work remains the compatibility baseline for whatever follows.
