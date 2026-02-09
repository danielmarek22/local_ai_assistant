
# Architecture Overview

This document describes how the Local AI Assistant works internally, focusing on **data flow**, **control flow**, and **responsibility boundaries**.

The goal is not only to explain *what* happens, but *why* the system is structured this way.

---

## High-Level Flow

At a high level, every user interaction follows this loop:

1. User input enters the system
2. Input is normalized and interpreted
3. Context is assembled (memory + state)
4. A planner decides what to do
5. Actions are executed (LLM, tools, services)
6. Results are stored in memory
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
- Routes planner decisions to execution

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

---

### LLM Layer (`app/llm`)

**Role:** Reasoning engine

The LLM layer:
- Loads and manages models
- Formats prompts
- Executes inference
- Returns structured outputs

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
- Stores conversations and facts
- Applies importance, decay, or summarization
- Retrieves relevant past information

Memory is treated as an **active subsystem**, not a passive database.

---

### Storage (`app/storage`)

**Role:** Persistence backend

Storage:
- Abstracts databases and files
- Ensures consistency
- Provides a clean API to memory and services

No other module touches the database directly.

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
- UI → Perception → Context assembly → Planner → Execution → Memory → UI

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
