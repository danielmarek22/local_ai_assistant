---
title: Getting started
description: Prepare and run a local Astra development environment.
---

# Getting started

This guide describes the repository's current development path. Astra expects local model and media services, so the exact runtime footprint depends on the enabled configuration.

## Prerequisites

- Python 3.12-compatible environment
- Ollama running locally
- A model compatible with the configured text, thinking, and multimodal features
- GPU/runtime dependencies required by the selected STT and TTS engines
- Optional services such as SearXNG or Mindcraft when their integrations are enabled

## Prepare configuration

The committed template lives at `app/config/assistant-template.yaml`. Create the ignored local file `app/config/assistant.yaml`, then adjust at least:

- `llm.host` and `llm.model`;
- generation and thinking options supported by that model;
- enabled integrations and their local endpoints;
- TTS and STT engines;
- assistant identity and prompt.

!!! warning "Local configuration is intentionally untracked"
    `app/config/assistant.yaml` can contain machine-specific paths and behavioral configuration. Do not replace the template with personal values.

## Install runtime dependencies

The complete local runtime preserves the original installation path:

```bash
venv_app/bin/python -m pip install -r requirements.txt
```

`requirements.txt` composes three exact-pinned groups:

- `requirements-core.txt` — web UI, configuration, validation, SQLite/vector storage,
  embeddings, vision-model support, and the remote GPT-SoVITS client;
- `requirements-media.txt` — selectable local PocketTTS, Piper, and Whisper backends;
- `requirements-integrations.txt` — the optional Mindcraft Socket.IO transport.

For a reduced installation, install `requirements-core.txt` plus only the optional
groups your configuration uses. A core-only server requires remote GPT-SoVITS, disabled
STT, and no Mindcraft transport. PyTorch wheels are
platform-specific; install the appropriate `torch==2.11.0` wheel first when the default
package index does not match the local CPU/CUDA environment.

## Run Astra

The browser experience is served by the FastAPI application. The console entry point in `main.py` is useful for narrower text-only experiments.

```bash
venv_app/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` after startup completes.

## Verify changes

Run the complete Python suite from the repository root:

```bash
venv_app/bin/python -m pip install -r requirements-test.txt
venv_app/bin/python -m unittest discover -s tests -v
```

Browser-side JavaScript tests use Node's built-in test runner and have no package
dependencies:

```bash
node --test tests/*.mjs
```

## Build these docs

Documentation has its own lightweight dependency set:

```bash
venv_app/bin/python -m pip install -r requirements-docs.txt
venv_app/bin/python -m mkdocs serve
```

The production-style local build is:

```bash
venv_app/bin/python -m mkdocs build --strict
```

Generated HTML is written to `site/` and is not committed.
