
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

- `resolve_think_value(think_override=None)` resolves the configured reasoning mode.
- `chat(...)` returns a completed assistant message for blocking helper calls.
- `chat_buffered(...)` consumes a complete generation before returning its message,
  including native tool calls. Interrupted generation must not expose a partial call.
- `stream_chat(...)` yields visible text, thinking markers, or native tool-call
  dictionaries as generation proceeds.

`options_override` is used for one-off generation settings without mutating the
client's default model options.

The complete signatures are defined in `base.py`. Blocking calls support timeout,
retry, tool, structured-format, and telemetry overrides. Buffered and visible
streaming calls support timeout and total-generation deadlines; buffered calls also
carry phase and iteration metadata. All inference methods accept session identity
for telemetry.

Core generation calls these interfaces directly. Backends and wrappers must implement
the declared methods and keyword arguments; unsupported legacy signatures are not
detected or retried at runtime. Ollama is the current production implementation.
An additional backend should adapt to this contract at construction and preserve
the distinction between atomic buffered results and visible streaming.

This layer should hide all backend-specific details from the rest of the system.

`image_fallback.py` owns the pure policy for recognizing image-related HTTP failures,
disabling transport retries for image requests, and generating progressively reduced
message candidates. `ollama_stream.py` remains responsible for HTTP and client state.
`thinking_filter.py` separates streamed reasoning markers from visible model output;
it is model-output normalization rather than turn orchestration.
