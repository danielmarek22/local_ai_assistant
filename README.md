# Local AI Assistant Backend

This project is a **fully local AI assistant backend** inspired by *Razer Project Ava*, but designed to run **entirely on the user’s machine**.

The focus is on **correct architecture**, **explicit control**, and **extensibility**, rather than prompt hacks or monolithic scripts. The system is designed to support future frontends such as **Live2D avatars**, **TTS**, and **real-time UI clients**, while remaining model- and backend-agnostic.

---

## High-Level Goals

- Run **fully locally** (no cloud dependencies)
- Support **stateful conversations**
- Separate **reasoning, planning, tools, and memory**
- Enable **tool use** (web search) in a controlled way
- Remain **model-agnostic** (Ollama now, llama.cpp later)
- Be inspectable, debuggable, and extensible

---

## Core Features

### ✅ Local LLM Inference
- Uses **Ollama** as the current backend
- Streaming responses via OpenAI-compatible API
- Easy to swap models (DeepSeek, LLaMA, Mistral, etc.)

### ✅ Persistent Chat History
- Stored in **SQLite**
- Session-based
- Lossless (no rewriting or deletion)
- Used only as a data source, never blindly injected

### ✅ Long-Term Memory
- Explicit memory writes (“remember this”)
- Stored separately from chat history
- Persisted across sessions
- Injected into context deliberately

### ✅ Context Builder (Central Control Point)
- Single place where prompts are assembled
- Injects:
  - system instructions
  - long-term memory
  - conversation summaries
  - web search context
  - recent user turns
- Prevents prompt bloat and accidental leakage

### ✅ Conversation Summarization
- Older conversations are summarized into a **factual, neutral recap**
- Summaries are:
  - domain-agnostic
  - non-advisory
  - safe against hallucination
- Prevents long-context degradation
- Full chat history is always preserved

### ✅ Planner (Hybrid)
- **Rule-based planner** for deterministic behavior
- **LLM-based planner** for flexible intent detection
- Unified decision format
- Planner decides *whether* to use tools, never executes them

### ✅ Local Web Search (SearXNG)
- Uses **SearXNG** running locally via Docker
- JSON API only (no HTML scraping)
- Privacy-friendly and offline-capable (LAN only)

### ✅ Constrained Search Result Summarization
- Raw search results are **never** shown to the main LLM
- A dedicated summarizer converts snippets into a neutral briefing
- Strong constraints prevent hallucinated facts or quantities
- Significantly improves factual grounding

---

## Architecture Overview
```
User Input
↓
Orchestrator
├─ Chat History Store
├─ Manual Memory Handler
├─ Planner (Hybrid)
│ ├─ Rule-based
│ └─ LLM-based
├─ Web Search (SearXNG, optional)
├─ Search Result Summarizer (optional)
├─ Context Builder
│ ├─ System Prompt
│ ├─ Long-Term Memory
│ ├─ Conversation Summary
│ ├─ Web Context
│ └─ Recent User Turns
↓
LLM (Streaming)
↓
Assistant Events
↓
History + Optional Summarization
```
---

## Project Structure
```
app/
├── core/
│ ├── orchestrator.py
│ ├── context_builder.py
│ ├── planner.py
│ ├── llm_planner.py
│ ├── hybrid_planner.py
│ └── summarizer.py
│
├── memory/
│ ├── chat_history.py
│ ├── memory_store.py
│ └── summary_store.py
│
├── storage/
│ └── database.py
│
├── tools/
│ ├── web_search.py
│ ├── search_formatter.py
│ └── search_summarizer.py
│
├── llm/
│ └── ollama_client.py
│
├── ui/
│ └── console.py
│
└── config/
└── assistant.yaml
```

---

## Configuration

All model and behavior configuration is externalized in YAML:

- Model name
- Backend host
- Generation parameters
- System prompt

This allows rapid iteration without touching code.

---

## Why This Design

This project deliberately avoids:
- prompt-only “agent” logic
- letting the LLM decide which tools exist
- stuffing full chat logs into the prompt
- hidden state or global magic

Instead, it follows a **backend-first philosophy**:
> The LLM is a component — not the system.

This makes the assistant:
- predictable
- debuggable
- extensible
- safe to evolve

---

## Planned Extensions

The backend is intentionally ready for:

- 3D avatar frontend
- TTS and audio I/O
- Emotional state signaling
- Tool caching
- Citation rendering
- llama.cpp backend swap
- Vision or multimodal inputs

No major refactor is required for these.

---

## Status

**Backend: feature-complete and stable**

The current focus can safely shift to:
- frontend integration
- UX and embodiment (Live2D)
- model experimentation
- performance tuning

---

## License

Local / personal use.  
License to be defined depending on future distribution.

