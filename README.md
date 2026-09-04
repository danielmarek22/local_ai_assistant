
# Local AI Assistant

Local AI Assistant is a modular, locally-runnable AI assistant framework designed for experimentation with LLMs, planning, memory, perception, and tool use.  
It emphasizes **hackability**, **clear separation of concerns**, and **offline / local-first execution**.

The current implementation is optimized for a **single local user** running the assistant on one machine.

## PSA: This is just a toy project and most of it was vibe-coded.

## High-level Architecture

The system is composed of several loosely-coupled subsystems:

- **Core orchestration** – per-turn coordination, action routing, logging
- **LLM interface** – model loading, prompting, and inference
- **Memory & storage** – conversation history, attachments, semantic memory, summaries, and persistence
- **Tools & services** – validated external capabilities exposed to the orchestrator
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
- current-turn images are sent to the multimodal model; historical images are represented by their stored summaries so prompt size does not grow with old binary payloads
- each stored image is summarized once for long-term retrieval through episodic vector search
- deleting a session removes its SQLite rows, vector entries, and uploaded files

In practice, this gives the project two views of memory:

- a structured local database for browsing, summaries, and session management
- a vector search layer for semantic retrieval during turns

The browser also includes a read-only Knowledge tab for inspecting effective beliefs,
all stored belief records, the exact belief section inserted into Astra's context, and
global structured memories saved through the memory tool. Summary and vector-search
inspection remain future work.

## Entry Points

- `main.py` – application entry point
- `app/server.py` – HTTP / UI server bootstrap
- `app/config/assistant.yaml` – main configuration file

## Philosophy

This project is intentionally **not** a polished product.  
It is a research playground for:

- Agent architectures
- Memory policies
- Capability routing and tool isolation
- Running LLMs efficiently on consumer hardware

Expect iteration, forks, and refactors.

## Directory Overview

See the `README.md` files inside each subdirectory of `app/` for detailed explanations.

## Documentation

The generated architecture and development documentation lives in `docs/` and is
configured by `mkdocs.yml`.

From the project root, install the lightweight documentation dependencies once:

```bash
venv_app/bin/python -m pip install -r requirements-docs.txt
```

Then start the live-reloading documentation server:

```bash
venv_app/bin/python -m mkdocs serve --dev-addr 127.0.0.1:8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in a browser. You can replace
`8000` with another free port when running several previews at once. Stop the server
with `Ctrl+C`.

To validate the complete site without starting a server, run:

```bash
venv_app/bin/python -m mkdocs build --strict
```

The generated HTML is written to the ignored `site/` directory. More maintenance
notes are available in the [documentation guide](docs/guides/documentation.md).

## Running Tests

The project includes a unit test suite under `tests/` covering context construction, orchestrator turn flow, image attachment persistence, session management, tool execution, and vector-backed memory behavior.
From the project root, run:

```bash
python -m unittest discover -s tests -v
```

For a deeper breakdown of test scope and how to add more coverage, see `tests/README.md`.
