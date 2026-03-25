
# Architecture Overview

This document describes how the Local AI Assistant works internally, focusing on **data flow**, **control flow**, and **responsibility boundaries**.

The goal is not only to explain *what* happens, but *why* the system is structured this way.

---

## High-Level Flow

At a high level, every user interaction follows this loop:

1. User input enters the system
2. Perception state is updated with the raw input
3. Relevant memory is retrieved from the vector database
4. The current turn is persisted to chat history
5. A planner decides what to do
6. Actions are executed (LLM, tools, services)
7. Context is assembled from system prompt, summaries, recent history, retrieved memory, and tool output
8. The assistant response is streamed and persisted
9. Conversation summarization may run
7. A response is returned to the user

This loop repeats until the interaction is complete.

---

## Core Components and Responsibilities

### UI Layer (`app/ui`)

**Role:** Human ↔ Machine boundary

The UI layer:
- Accepts user input
- Displays or streams responses
- Contains *no* business logic

It forwards raw input directly into the core orchestration layer.

---

### Perception Layer (`app/perception`)

**Role:** Raw input → structured signal

Responsibilities:
- Normalize text
- Extract basic intent or signals
- Prepare a clean representation for planners

This layer intentionally stays lightweight.  
Heavy reasoning belongs elsewhere.

---

### Core Orchestrator (`app/core`)

**Role:** Central nervous system

The orchestrator:
- Maintains session state
- Coordinates all subsystems
- Runs the main decision loop
- Retrieves semantic and episodic memory
- Routes planner decisions to execution
- Streams assistant output back as events
- Triggers summarization when a session reaches the configured threshold

Nothing else in the system should be aware of the *entire* system state.

---

### Planner (`app/planners`)

**Role:** Decision-making

Planners:
- Observe current context (input + memory + state)
- Decide on the next action(s)
- Choose between:
  - LLM calls
  - Tool invocations
  - Memory writes
  - Passive responses

Planners **do not execute actions themselves** — they only describe intent.

This separation makes planner logic easy to experiment with and replace.

Current planner modes:
- **Rule planner** for lightweight heuristics
- **LLM planner** for JSON action planning
- **Hybrid planner** to let rules handle obvious intents first

---

### LLM Layer (`app/llm`)

**Role:** Reasoning engine

The LLM layer:
- Loads and manages models
- Formats prompts
- Executes inference
- Returns structured outputs
- Supports per-request option overrides for non-streaming calls used by planners and summarizers

All backend-specific details (local vs API, quantization, batching) live here.

---

### Tools (`app/tools`)

**Role:** Acting on the world

Tools:
- Are explicitly callable actions
- Have clear inputs and outputs
- Are deterministic and scoped

Examples:
- File system access
- System queries
- External APIs

Tools never decide *when* they are used — planners do.

---

### Memory System (`app/memory`)

**Role:** Persistence and continuity

Memory:
- Stores conversations, long-term facts, and session summaries
- Applies a lightweight memory policy for explicit memory writes
- Retrieves relevant past information through vector search
- Separates **semantic memory** from **episodic conversation memory**

Memory is treated as an **active subsystem**, not a passive database.

Current implementation details:
- `MemoryStore` writes long-term fact-like memory to both SQLite and the Chroma `semantic_memory` collection.
- `ChatHistoryStore` writes every message to both SQLite and the Chroma `episodic_memory` collection.
- `SummaryStore` keeps one rolling summary per session in SQLite.
- When a session is deleted, both SQLite rows and episodic vector entries for that session are removed.

---

### Storage (`app/storage`)

**Role:** Persistence backend

Storage:
- Owns the SQLite connection and schema initialization
- Provides a shared `Database` object used by stores
- Provides a local Chroma vector store used for semantic retrieval
- Centralizes persistence configuration (path, connection setup)

Current implementation note:
- `Database` owns the SQLite schema for `chat_history`, `memory`, and `conversation_summary`.
- `VectorStore` owns two Chroma collections:
  - `semantic_memory`
  - `episodic_memory`
- Store classes in `app/memory` coordinate both layers directly.
- So the practical boundary is:
  - `core -> memory stores -> sqlite/chroma`
  - rather than a strict repository abstraction layer.

---

### Services (`app/services`)

**Role:** Background processes

Services:
- Run alongside the main loop
- Perform monitoring or maintenance
- Can influence state indirectly

They are persistent, unlike tools.

---

## Control vs Data Flow

**Control Flow**
- Orchestrator → Planner → Action dispatch

**Data Flow**
- UI → Perception → Vector retrieval → Planner → Tool/memory actions → Context assembly → LLM → Persistence → UI

Keeping these separate reduces coupling and cognitive load.

---

## Design Principles

- Clear separation of concerns
- Planner describes *intent*, not *execution*
- Memory is first-class
- Everything is replaceable
- Optimized for local execution and experimentation

---

## Why This Matters

This architecture makes it easy to:
- Swap planners
- Experiment with memory policies
- Add new tools safely
- Run on limited hardware
- Treat the assistant as a research system, not a black box

If something feels hard to change, it probably violates one of these principles.
