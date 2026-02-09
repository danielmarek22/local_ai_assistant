
# Local AI Assistant

Local AI Assistant is a modular, locally-runnable AI assistant framework designed for experimentation with LLMs, planning, memory, perception, and tool use.  
It emphasizes **hackability**, **clear separation of concerns**, and **offline / local-first execution**.

## PSA: Most of this project was vibe-coded. 

## High-level Architecture

The system is composed of several loosely-coupled subsystems:

- **Core orchestration** – session lifecycle, action routing, logging
- **LLM interface** – model loading, prompting, and inference
- **Planners** – decide *what* the assistant should do next
- **Memory & storage** – long-term and short-term memory management
- **Tools & services** – external capabilities exposed to the planner
- **UI & server** – user interaction layer

Each major subsystem lives in its own directory under `app/` and is documented individually.

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
