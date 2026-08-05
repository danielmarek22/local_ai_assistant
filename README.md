
# Local AI Assistant

Local AI Assistant is a modular, locally-runnable AI assistant framework designed for experimentation with LLMs, planning, memory, perception, and tool use.  
It emphasizes **hackability**, **clear separation of concerns**, and **offline / local-first execution**.

The current implementation is optimized for a **single local user** running the assistant on one machine.

## PSA: This is just a toy project and most of it was vibe-coded.

## High-level Architecture

The system is composed of several loosely-coupled subsystems:

- **Core orchestration** – per-turn coordination, action routing, logging
- **LLM interface** – model loading, prompting, and inference
- **Planners** – decide *what* the assistant should do next
- **Memory & storage** – conversation history, attachments, semantic memory, summaries, and persistence
- **Tools & services** – external capabilities exposed to the planner
- **UI & server** – user interaction layer

Each major subsystem lives in its own directory under `app/` and is documented individually.

## Current Memory Model

The assistant currently uses a **hybrid memory system**:

- **SQLite** stores canonical records for chat history, attachment metadata, long-term memory rows, and per-session summaries.
- **ChromaDB** stores embeddings for:
  - semantic memory (`semantic_memory`)
  - episodic conversation history and image summaries (`episodic_memory`)
- The orchestrator retrieves:
  - relevant long-term facts from semantic memory
  - relevant messages and stored image summaries from past sessions via episodic memory
- Retrieved memory and optional passive integration state are injected into the prompt as background context before response generation.

Attachments are handled as a first-class part of the conversation. The current implementation supports images and keeps a lightweight base `Attachment` model so other types can be added later without reshaping the turn pipeline:

- the browser converts selected or pasted images to Base64 before sending them
- the backend stores the image bytes on disk and attachment metadata in SQLite
- recent image turns can be replayed directly into the multimodal prompt
- each stored image is summarized once for long-term retrieval through episodic vector search
- deleting a session removes its SQLite rows, vector entries, and uploaded files

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

The project includes a unit test suite under `tests/` covering planner logic, context construction, orchestrator turn flow, image attachment persistence, session management, tool execution, and vector-backed memory behavior.  
From the project root, run:

```bash
python -m unittest discover -s tests -v
```

For a deeper breakdown of test scope and how to add more coverage, see `tests/README.md`.
