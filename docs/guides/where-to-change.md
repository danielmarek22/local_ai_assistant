---
title: Where do I change...?
description: A task-first lookup for finding Astra's configuration, implementation, and tests.
---

# Where do I change...?

Start with the narrowest row matching your goal. Paths are relative to the repository root.

| I want to change... | Configuration or content | Implementation | Tests and notes |
| --- | --- | --- | --- |
| The model, endpoint, or generation settings | `app/config/assistant.yaml` → `llm` | `app/llm/ollama_stream.py` | `tests/test_http_retries.py`, orchestrator tests |
| What Astra sees in a prompt | `context`, `orchestrator`, and `beliefs` sections | `app/core/context_builder.py` | `tests/test_context_builder.py` |
| How restored chats regain context | `context.history_limit` | `app/memory/chat_history.py`, `app/core/context_builder.py`, session routes in `app/server.py` | `tests/test_chat_history.py`, `tests/test_server_sessions.py` |
| Conversation summaries | `orchestrator.summary_trigger` | `app/memory/history_summarizer.py`, `app/memory/summary_store.py`, `app/core/turn_finalizer.py` | Context and orchestrator tests |
| Long-term memory retrieval | `context.injected_memory_limit`, `integrations.memory` | `app/memory/retriever.py`, `app/memory/memory_store.py`, `app/storage/vector_store.py` | `tests/test_memory_store.py`, `tests/test_memory_reflector.py` |
| Image or audio attachment handling | `voice_input` for spoken audio | `app/perception/attachments.py`, `app/perception/frame_controller.py`, `app/core/turn_input.py`, `app/server.py` | `tests/test_attachment_utils.mjs`, `tests/test_perception_frames.py`, `tests/test_turn_input.py` |
| The system prompt or assistant name | `assistant` | Loaded through `app/config.py` | `tests/test_config.py` |
| Speech recognition | `stt` | `app/stt/factory.py`, `app/stt/whisper_engine.py` | Exercise a short microphone recording |
| Astra's voice | `tts` | `app/tts/factory.py` and the selected engine | See [PocketTTS](../services/pocket-tts.md) |
| Tool routing and execution | `integrations` | `app/core/tool_executor.py`, `app/integrations/registry.py`, `app/core/orchestrator.py` | `tests/test_tool_executor.py`, `tests/test_integration_registry.py` |
| WebSocket framing, size limits, or approval waits | — | `app/transport/websocket_protocol.py`, `app/transport/websocket_connection.py` | `tests/test_websocket_protocol.py`, `tests/test_server_sessions.py` |
| Screen/webcam frame handling or vision watchdog events | — | `app/perception/frame_controller.py`, `app/perception/vision_watchdog.py` | `tests/test_perception_frames.py` |
| Web search | `integrations.web` | `app/tools/web_search.py`, `app/integrations/builtins.py` | `tests/test_builtin_integrations.py` |
| Shell command behavior | `integrations.shell` | `app/tools/bash_execution.py` | `tests/test_bash_execution.py` |
| Belief extraction or display | `beliefs` | `app/beliefs/`, `app/knowledge/` | `tests/test_beliefs.py`, `tests/test_belief_react.py`, `tests/test_knowledge.py` |
| Autonomous reactions | `autonomy` | `app/autonomy/` | `tests/test_autonomy_broker.py`, `tests/test_autonomy_coordination.py` |
| Minecraft behavior | `integrations.mindcraft` plus the fork's `settings.js` and profile | `app/integrations/mindcraft.py` | [Mindcraft setup](../services/mindcraft.md), `tests/test_mindcraft_integration.py` |
| Avatar expressions or outfits | `assistant.avatar_controls` | `app/avatar/controls.py`, `app/avatar/stream_processor.py`, `app/integrations/outfit.py`, browser assets | Avatar and outfit tests |
| Browser behavior or appearance | — | `static/` | Relevant `tests/*.mjs` files |
| Logging and traces | `logging` | `app/logging.py` | `tests/test_logging.py` |
| Documentation navigation or theme | `mkdocs.yml` | `docs/`, `docs/assets/stylesheets/astra.css` | `venv_app/bin/python -m mkdocs build --strict` |

## A safe change loop

1. Reproduce the behavior and identify the smallest owning boundary.
2. Add or locate a test close to that boundary.
3. Change the implementation without moving authority into model output or browser state.
4. Run the narrow test, then the broader suite if the boundary is shared.
5. Update this documentation when configuration meaning or system flow changed.

For how those boundaries connect, return to the [Project map](../project-map.md).
