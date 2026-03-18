
# Local AI Assistant

Local AI Assistant is a modular, locally-runnable AI assistant framework designed for experimentation with LLMs, planning, memory, perception, and tool use.  
It emphasizes **hackability**, **clear separation of concerns**, and **offline / local-first execution**.

The current implementation is optimized for a **single local user** running the assistant on one machine.

## PSA: This is just a toy project and most of it was vibe-coded.

## High-level Architecture

The system is composed of several loosely-coupled subsystems:

- **Core orchestration** – session lifecycle, action routing, logging
- **LLM interface** – model loading, prompting, and inference
- **Planners** – decide *what* the assistant should do next
- **Memory & storage** – conversation history, semantic memory, summaries, and persistence
- **Tools & services** – external capabilities exposed to the planner
- **UI & server** – user interaction layer

Each major subsystem lives in its own directory under `app/` and is documented individually.

## Current Memory Model

The assistant currently uses a **hybrid memory system**:

- **SQLite** stores canonical records for chat history, long-term memory rows, and per-session summaries.
- **ChromaDB** stores embeddings for:
  - semantic memory (`semantic_memory`)
  - episodic conversation history (`episodic_memory`)
- The orchestrator retrieves:
  - relevant long-term facts from semantic memory
  - relevant messages from past sessions via episodic memory
- Retrieved memory and tool output are injected into the prompt as background context before response generation.

In practice, this gives the project two views of memory:

- a structured local database for browsing, summaries, and session management
- a vector search layer for semantic retrieval during turns

## Entry Points

- `main.py` – application entry point
- `app/server.py` – HTTP / UI server bootstrap
- `app/config/assistant.yaml` – main configuration file

## Philosophy

This project is intentionally **not** a polished product.  
It is a research playground for:

- Agent architectures
- Memory policies
- Planner / tool separation
- Running LLMs efficiently on consumer hardware

Expect iteration, forks, and refactors.

## Directory Overview

See the `README.md` files inside each subdirectory of `app/` for detailed explanations.

## Running Tests

The project includes a unit test suite under `tests/` covering planner logic, context construction, orchestrator turn flow, retry behavior, session management, tool execution, and vector-backed memory behavior.  
From the project root, run:

```bash
python -m unittest discover -s tests -v
```

For a deeper breakdown of test scope and how to add more coverage, see `tests/README.md`.
