
# LLM

Large Language Model integration layer.

## Responsibilities
- Model loading and initialization
- Prompt formatting
- Inference calls
- Abstraction over different backends (local models, APIs)
- Per-request inference overrides for structured helper calls
- Backend-specific image capability detection and deterministic fallback policy

## Current Interface

The shared `LLMClient` contract currently exposes:

- `chat(messages, think_override=None, options_override=None)`
- `stream_chat(messages, think_override=None)`

`options_override` is used for one-off generation settings without mutating the
client's default model options.

This layer should hide all backend-specific details from the rest of the system.

`image_fallback.py` owns the pure policy for recognizing image-related HTTP failures,
disabling transport retries for image requests, and generating progressively reduced
message candidates. `ollama_stream.py` remains responsible for HTTP and client state.
