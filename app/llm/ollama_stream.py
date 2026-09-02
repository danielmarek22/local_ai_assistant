import requests
import json
import time
import logging
from typing import Iterator

from .base import InferenceFailure, LLMClient
from app.core.thinking_filter import ThinkingBlockSplitter
from app.logging import trace_event

logger = logging.getLogger("ollama_client")

# Default repetition penalty applied when the caller has not set one.
_DEFAULT_REPEAT_PENALTY = 1.15

# Maximum temperature allowed for streaming responses. If the configured
# value exceeds this cap it is clamped down, not replaced with a hardcoded 1.
_MAX_STREAM_TEMPERATURE = 1.5

# Minimum context / prediction window sizes used when thinking is active.
_THINKING_MIN_CTX = 8192
_THINKING_MIN_PREDICT = 2048

# Stop sequences appended unconditionally to every streaming request to
# prevent the model from rambling past a natural end-of-turn token.
#
# NOTE: The native /api/chat endpoint does not support OpenAI-style `stop`
# sequences at the top level; they must be passed inside `options`. See
# _build_stream_options for where these are injected.
_BUILTIN_STOP_SEQUENCES = ["<|eot_id|>", "<|im_end|>", "<|end_of_sentence|>"]
_MAX_BUFFERED_LINE_BYTES = 512_000
_MAX_BUFFERED_CONTENT_CHARS = 256_000
_MAX_BUFFERED_THINKING_CHARS = 1_000_000
_MAX_BUFFERED_TOOL_JSON_CHARS = 128_000


class OllamaClient(LLMClient):
    """
    Ollama-backed LLM client using the native /api/chat endpoint exclusively.

    Supports both blocking (chat) and streaming (stream_chat) call patterns,
    as well as multimodal (image) inputs.  Thinking tokens (extended
    reasoning) are an opt-in feature controlled by `thinking_enabled` and
    `thinking_level`.

    Call `preload()` after construction if you want the model weights loaded
    into VRAM before the first real request. This is intentionally not done
    in __init__ to keep construction side-effect-free and easy to test.
    """

    def __init__(
        self,
        model: str,
        host: str,
        options: dict | None = None,
        thinking_enabled: bool = False,
        thinking_level: str | None = None,
        thinking_options: dict | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.25,
    ):
        self.model = model
        self._preload_url = f"{host}/api/generate"
        self.url = f"{host}/api/chat"
        self.options = options or {}
        self.thinking_enabled = thinking_enabled
        self.thinking_level = thinking_level
        self.thinking_options = thinking_options or {}
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._multimodal_supported = True
        self.last_chat_dropped_current_images = False
        self.last_chat_dropped_current_images_count = 0
        self.last_chat_image_fallback_strategy: str | None = None
        self.last_stream_dropped_current_images = False
        self.last_stream_dropped_current_images_count = 0
        self.last_stream_image_fallback_strategy: str | None = None

        # Reuse a single TCP connection across all requests.
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """
        Warm the model into VRAM so the first real request has no cold-start
        latency. Call this explicitly after construction when desired; it is
        NOT called automatically in __init__.
        """
        logger.info("Preloading %s into VRAM…", self.model)
        try:
            self.session.post(
                self._preload_url,
                json={"model": self.model, "keep_alive": "-1"},
                timeout=30,
            )
            logger.info("Model preloaded successfully.")
        except Exception as exc:
            logger.warning("Failed to preload model: %s", exc)

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages,
        think_override=None,
        options_override: dict | None = None,
        timeout_override: float | None = None,
        max_retries_override: int | None = None,
        tools: list[dict] | None = None,
        format_override: dict | str | None = None,
    ) -> dict:
        """
        Blocking, non-streaming call.
        Used for planners, summarizers, and other structured outputs.

        `options_override` is merged on top of instance defaults per-call
        without mutating them. `stream_chat` intentionally does not expose
        this parameter because streaming responses are always user-facing and
        apply their own fixed safety defaults (temperature cap, stop tokens).
        """
        think_value = self._resolve_think_value(think_override)

        request_options = self.options.copy()
        if options_override:
            request_options.update(options_override)

        request_messages = messages
        self.last_chat_dropped_current_images = False
        self.last_chat_dropped_current_images_count = 0
        self.last_chat_image_fallback_strategy = None

        if not self._multimodal_supported:
            request_messages, _ = self._strip_images_from_messages(request_messages)

        payload = self._build_payload(
            request_messages,
            stream=False,
            think_value=think_value,
            options=request_options,
            tools=tools,
            format_override=format_override,
        )

        logger.info(
            "Ollama chat request (stream=False, think=%r, messages=%d)",
            think_value,
            len(messages),
        )
        trace_event("llm", "chat_request", payload=payload)

        request_started = time.perf_counter()
        try:
            r = self._post_with_retry(
                payload,
                stream=False,
                timeout_override=timeout_override,
                max_retries_override=self._resolve_image_request_retries(
                    request_messages,
                    max_retries_override,
                ),
            )
        except requests.HTTPError as exc:
            if not self._should_retry_without_images(exc, request_messages):
                raise self._inference_failure(exc) from exc

            response_text = self._http_error_text(exc)
            if self._error_indicates_model_without_images(response_text):
                self._multimodal_supported = False

            fallback_candidates = self._build_image_fallback_messages(request_messages)
            if not fallback_candidates:
                raise

            last_exc: requests.HTTPError = exc
            for fallback_messages, strategy, dropped_current_images_count in fallback_candidates:
                payload["messages"] = fallback_messages
                logger.warning(
                    "Ollama chat rejected image payload (status=%s); retrying %s.",
                    getattr(last_exc.response, "status_code", None),
                    strategy,
                )
                trace_event(
                    "llm",
                    "chat_retry_without_images",
                    payload={
                        "status_code": getattr(last_exc.response, "status_code", None),
                        "response_text": self._http_error_text(last_exc),
                        "strategy": strategy,
                        "dropped_current_images_count": dropped_current_images_count,
                    },
                )
                try:
                    r = self._post_with_retry(
                        payload,
                        stream=False,
                        timeout_override=timeout_override,
                        max_retries_override=self._resolve_image_request_retries(
                            fallback_messages,
                            max_retries_override,
                        ),
                    )
                    self.last_chat_image_fallback_strategy = strategy
                    if dropped_current_images_count > 0:
                        self.last_chat_dropped_current_images = True
                        self.last_chat_dropped_current_images_count = dropped_current_images_count
                    break
                except requests.HTTPError as retry_exc:
                    if not self._should_retry_without_images(retry_exc, fallback_messages):
                        raise self._inference_failure(retry_exc) from retry_exc
                    last_exc = retry_exc
                    if self._error_indicates_model_without_images(self._http_error_text(retry_exc)):
                        self._multimodal_supported = False
                except requests.RequestException as retry_exc:
                    raise self._inference_failure(retry_exc) from retry_exc
            else:
                raise self._inference_failure(last_exc) from last_exc
        except requests.RequestException as exc:
            raise self._inference_failure(exc) from exc

        try:
            data = r.json()
        except (TypeError, ValueError) as exc:
            raise InferenceFailure(
                "malformed_response",
                "Ollama returned malformed JSON for a completed generation.",
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("message", {}), dict):
            raise InferenceFailure(
                "malformed_response",
                "Ollama returned an invalid completed-generation response shape.",
            )
        message = data.get("message", {})
        finish_reason = data.get("done_reason")

        trace_event(
            "llm",
            "chat_response",
            payload={
                "content": message.get("content"),
                "thinking": message.get("thinking"),
                "done_reason": finish_reason,
                "http_wall_duration_ms": round(
                    (time.perf_counter() - request_started) * 1000, 2
                ),
                "ollama_total_duration_ms": self._nanoseconds_to_ms(data.get("total_duration")),
                "ollama_load_duration_ms": self._nanoseconds_to_ms(data.get("load_duration")),
                "ollama_prompt_eval_duration_ms": self._nanoseconds_to_ms(
                    data.get("prompt_eval_duration")
                ),
                "ollama_generation_duration_ms": self._nanoseconds_to_ms(
                    data.get("eval_duration")
                ),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "generation_token_count": data.get("eval_count"),
            },
        )

        return message

    def chat_buffered(
        self,
        messages,
        think_override=None,
        options_override: dict | None = None,
        timeout_override: float | None = None,
        generation_deadline_s: float | None = 120.0,
        tools: list[dict] | None = None,
        generation_phase: str | None = None,
        react_iteration: int | None = None,
    ) -> dict:
        """Consume a native streamed generation atomically before exposing its result."""
        think_value = self._resolve_think_value(think_override)
        request_options = self._build_stream_options(think_value)
        request_options = self._merge_options(request_options, options_override)
        payload = self._build_payload(
            messages,
            stream=True,
            think_value=think_value,
            options=request_options,
            tools=tools,
        )
        logger.info(
            "Ollama buffered chat request (stream=True, think=%r, messages=%d)",
            think_value,
            len(messages),
        )
        effective_num_predict = request_options.get("num_predict")
        trace_event("llm", "chat_request", payload={
            "buffered": True,
            "generation_phase": generation_phase,
            "react_iteration": react_iteration,
            "effective_think": think_value,
            "effective_num_predict": effective_num_predict,
            "tools_exposed": bool(tools),
            "tool_count": len(tools or []),
            "message_count": len(messages),
        })

        started = time.perf_counter()
        first_chunk_ms = None
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list = []
        content_chars = 0
        thinking_chars = 0
        completed = False
        final_chunk: dict = {}
        try:
            with self._post_stream(payload, timeout_override=timeout_override) as response:
                for line in response.iter_lines():
                    now = time.perf_counter()
                    if generation_deadline_s is not None and now - started > generation_deadline_s:
                        raise InferenceFailure(
                            "generation_deadline",
                            "Ollama buffered generation exceeded its total deadline.",
                        )
                    if not line:
                        continue
                    if first_chunk_ms is None:
                        first_chunk_ms = round((now - started) * 1000, 2)
                    if len(line) > _MAX_BUFFERED_LINE_BYTES:
                        raise InferenceFailure(
                            "malformed_response",
                            "Ollama stream chunk exceeded the bounded line size.",
                        )
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise InferenceFailure(
                            "malformed_response",
                            "Ollama returned malformed streamed JSON.",
                        ) from exc
                    if not isinstance(chunk, dict) or not isinstance(chunk.get("message", {}), dict):
                        raise InferenceFailure(
                            "malformed_response",
                            "Ollama returned an invalid streamed response shape.",
                        )
                    message = chunk.get("message", {})
                    content = message.get("content") or ""
                    thinking = message.get("thinking") or ""
                    if not isinstance(content, str) or not isinstance(thinking, str):
                        raise InferenceFailure(
                            "malformed_response",
                            "Ollama streamed non-text content or thinking data.",
                        )
                    content_chars += len(content)
                    thinking_chars += len(thinking)
                    if content_chars > _MAX_BUFFERED_CONTENT_CHARS:
                        raise InferenceFailure(
                            "malformed_response", "Ollama content exceeded the bounded buffer."
                        )
                    if thinking_chars > _MAX_BUFFERED_THINKING_CHARS:
                        raise InferenceFailure(
                            "malformed_response", "Ollama thinking exceeded the bounded buffer."
                        )
                    content_parts.append(content)
                    thinking_parts.append(thinking)
                    chunk_tool_calls = message.get("tool_calls")
                    if chunk_tool_calls is not None:
                        if not isinstance(chunk_tool_calls, list):
                            raise InferenceFailure(
                                "malformed_response", "Ollama streamed invalid tool-call data."
                            )
                        tool_calls.extend(chunk_tool_calls)
                        if len(json.dumps(tool_calls, ensure_ascii=True)) > _MAX_BUFFERED_TOOL_JSON_CHARS:
                            raise InferenceFailure(
                                "malformed_response", "Ollama tool calls exceeded the bounded buffer."
                            )
                    if chunk.get("done"):
                        completed = True
                        final_chunk = chunk
                        break
        except InferenceFailure:
            raise
        except requests.RequestException as exc:
            raise self._inference_failure(exc) from exc
        except Exception as exc:
            raise InferenceFailure(
                "malformed_response",
                f"Ollama buffered stream failed: {type(exc).__name__}",
            ) from exc

        if not completed:
            raise InferenceFailure(
                "malformed_response",
                "Ollama stream ended before a completed generation marker.",
            )
        message = {
            "content": "".join(content_parts),
            "thinking": "".join(thinking_parts),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        trace_event(
            "llm",
            "chat_response",
            payload={
                "buffered": True,
                "generation_phase": generation_phase,
                "react_iteration": react_iteration,
                "effective_think": think_value,
                "effective_num_predict": effective_num_predict,
                "tools_exposed": bool(tools),
                "tool_count": len(tools or []),
                "tool_call_count": len(tool_calls),
                "content_chars": len(message["content"]),
                "thinking_chars_received": min(
                    len(message["thinking"]), _MAX_BUFFERED_THINKING_CHARS
                ),
                "thinking_token_count": final_chunk.get("thinking_count"),
                "done_reason": final_chunk.get("done_reason"),
                "time_to_first_chunk_ms": first_chunk_ms,
                "total_duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "ollama_load_duration_ms": self._nanoseconds_to_ms(final_chunk.get("load_duration")),
                "ollama_prompt_eval_duration_ms": self._nanoseconds_to_ms(
                    final_chunk.get("prompt_eval_duration")
                ),
                "prompt_eval_count": final_chunk.get("prompt_eval_count"),
                "ollama_generation_duration_ms": self._nanoseconds_to_ms(
                    final_chunk.get("eval_duration")
                ),
                "generation_token_count": final_chunk.get("eval_count"),
            },
        )
        return message

    def stream_chat(
        self, 
        messages, 
        think_override=None, 
        tools: list[dict] | None = None,
        options_override: dict | None = None,
        generation_deadline_s: float | None = None,
        timeout_override: float | None = None,
    ) -> Iterator[str | dict]:
        """
        Streaming call. Yields text chunks for user-facing responses.

        Thinking tokens are wrapped in <think>…</think> and yielded inline
        so the orchestrator can process or strip them downstream.
        """
        think_value = self._resolve_think_value(think_override)
        request_options = self._build_stream_options(think_value)
        request_options = self._merge_options(request_options, options_override)
        request_messages = messages
        self.last_stream_dropped_current_images = False
        self.last_stream_dropped_current_images_count = 0
        self.last_stream_image_fallback_strategy = None

        if not self._multimodal_supported:
            request_messages, _ = self._strip_images_from_messages(request_messages)

        payload = self._build_payload(
            request_messages,
            stream=True,
            think_value=think_value,
            options=request_options,
            tools=tools,
        )

        logger.info(
            "Ollama chat request (stream=True, think=%r, messages=%d)",
            think_value,
            len(messages),
        )
        trace_event("llm", "stream_request", payload=payload)

        collected_content: list[str] = []
        collected_thinking: list[str] = []
        try:
            yield from self._stream_payload(
                payload,
                collected_content,
                collected_thinking,
                generation_deadline_s=generation_deadline_s,
                timeout_override=timeout_override,
            )
        except requests.HTTPError as exc:
            if not self._should_retry_without_images(exc, request_messages):
                raise self._inference_failure(exc) from exc

            response_text = self._http_error_text(exc)
            if self._error_indicates_model_without_images(response_text):
                self._multimodal_supported = False

            fallback_candidates = self._build_image_fallback_messages(request_messages)
            if not fallback_candidates:
                raise

            last_exc: requests.HTTPError = exc
            for fallback_messages, strategy, dropped_current_images_count in fallback_candidates:
                payload["messages"] = fallback_messages
                logger.warning(
                    "Ollama stream rejected image payload (status=%s); retrying %s.",
                    getattr(last_exc.response, "status_code", None),
                    strategy,
                )
                trace_event(
                    "llm",
                    "stream_retry_without_images",
                    payload={
                        "status_code": getattr(last_exc.response, "status_code", None),
                        "response_text": self._http_error_text(last_exc),
                        "strategy": strategy,
                        "dropped_current_images_count": dropped_current_images_count,
                    },
                )
                try:
                    yield from self._stream_payload(
                        payload, collected_content, collected_thinking,
                        generation_deadline_s=generation_deadline_s,
                        timeout_override=timeout_override,
                    )
                    self.last_stream_image_fallback_strategy = strategy
                    if dropped_current_images_count > 0:
                        self.last_stream_dropped_current_images = True
                        self.last_stream_dropped_current_images_count = dropped_current_images_count
                    break
                except requests.HTTPError as retry_exc:
                    if not self._should_retry_without_images(retry_exc, fallback_messages):
                        raise self._inference_failure(retry_exc) from retry_exc
                    last_exc = retry_exc
                    if self._error_indicates_model_without_images(self._http_error_text(retry_exc)):
                        self._multimodal_supported = False
            else:
                raise self._inference_failure(last_exc) from last_exc
        except requests.RequestException as exc:
            raise self._inference_failure(exc) from exc

        logger.info(
            "Ollama stream complete (content_len=%d, thinking_len=%d)",
            len("".join(collected_content)),
            len("".join(collected_thinking)),
        )
        trace_event(
            "llm",
            "stream_response",
            payload={
                "content": "".join(collected_content),
                "thinking": "".join(collected_thinking),
            },
        )

    # ------------------------------------------------------------------
    # Private — streaming
    # ------------------------------------------------------------------

    def _stream_payload(
        self,
        payload: dict,
        collected_content: list[str],
        collected_thinking: list[str],
        generation_deadline_s: float | None = None,
        timeout_override: float | None = None,
    ) -> Iterator[str | dict]: # Update return type
        """
        Consume a streaming response from /api/chat (native NDJSON format).
        """
        in_thinking_block = False

        started = time.perf_counter()
        with self._post_stream(payload, timeout_override=timeout_override) as r:
            for line in r.iter_lines():
                if generation_deadline_s is not None and time.perf_counter() - started > generation_deadline_s:
                    raise InferenceFailure(
                        "generation_deadline",
                        "Ollama streamed generation exceeded its total deadline.",
                    )
                if not line:
                    continue

                chunk = json.loads(line.decode("utf-8"))
                message = chunk.get("message", {})
                content = message.get("content")
                thinking = message.get("thinking")
                
                # Intercept native tool calls and yield as a dict
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    yield {"tool_calls": tool_calls}

                if thinking:
                    if not in_thinking_block:
                        yield "<think>\n"
                        in_thinking_block = True
                    collected_thinking.append(thinking)
                    yield thinking

                if content and in_thinking_block:
                    yield "\n</think>\n\n"
                    in_thinking_block = False

                if content:
                    collected_content.append(content)
                    yield content

                if chunk.get("done"):
                    if in_thinking_block:
                        yield "\n</think>\n\n"
                    if chunk.get("done_reason") == "length":
                        logger.warning(
                            "Ollama stream hit the token limit (num_predict) "
                            "before finishing — response may be truncated."
                        )
                    break

    # ------------------------------------------------------------------
    # Private — request builders
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages,
        stream: bool,
        think_value,
        options: dict | None = None,
        tools: list[dict] | None = None,
        format_override: dict | str | None = None,
    ) -> dict:
        """
        Build a request payload for the native /api/chat endpoint.
        """
        if tools and format_override is not None:
            raise ValueError("Ollama tools and format_override cannot be combined")

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": think_value,
        }
        if options:
            payload["options"] = self._normalize_options(options)
            
        # Natively inject schemas if provided
        if tools:
            payload["tools"] = tools
        if format_override is not None:
            payload["format"] = format_override
            
        return payload

    def _build_stream_options(self, think_value) -> dict:
        """
        Build the options dict for a streaming request.

        Applies several safety defaults on top of the instance options:
        - Applies thinking-specific overrides when thinking is active.
        - Renames max_tokens → num_predict for the native API.
        - Clamps temperature to _MAX_STREAM_TEMPERATURE if it exceeds it.
        - Sets a default repeat_penalty if none is configured.
        - Appends built-in stop sequences inside options.
        - Boosts context / prediction limits when thinking is active.
        """
        opts = self.options.copy()
        if think_value:
            opts = self._merge_options(opts, self.thinking_options)

        opts = self._drop_none_options(opts)

        # Native endpoint uses num_predict, not max_tokens.
        if "max_tokens" in opts and "num_predict" not in opts:
            opts["num_predict"] = opts.pop("max_tokens")

        # Default repetition penalty to discourage looping.
        opts.setdefault("repeat_penalty", _DEFAULT_REPEAT_PENALTY)

        # Clamp temperature — do not silently replace a deliberate value,
        # just bring it down to the ceiling if it exceeds it.
        current_temp = opts.get("temperature")
        if current_temp is None or current_temp > _MAX_STREAM_TEMPERATURE:
            opts["temperature"] = _MAX_STREAM_TEMPERATURE

        # Merge stop sequences rather than replacing caller-supplied ones.
        existing_stops = opts.get("stop", [])
        if isinstance(existing_stops, str):
            existing_stops = [existing_stops]
        opts["stop"] = existing_stops + _BUILTIN_STOP_SEQUENCES

        # Expand context window when thinking is active.
        if think_value:
            opts["num_ctx"] = max(opts.get("num_ctx", _THINKING_MIN_CTX), _THINKING_MIN_CTX)
            opts["num_predict"] = max(
                opts.get("num_predict", _THINKING_MIN_PREDICT), _THINKING_MIN_PREDICT
            )

        return opts

    def _normalize_options(self, options: dict) -> dict:
        """
        Normalise an options dict for the native endpoint:
        - Drop None values.
        - Rename max_tokens → num_predict.
        """
        normalized = self._drop_none_options(options)
        if "max_tokens" in normalized and "num_predict" not in normalized:
            normalized["num_predict"] = normalized.pop("max_tokens")
        return normalized

    # ------------------------------------------------------------------
    # Private — HTTP
    # ------------------------------------------------------------------

    def _post_stream(self, payload: dict, timeout_override: float | None = None) -> requests.Response:
        """
        Issue a streaming POST to the native chat endpoint. Returns the raw
        Response used as a context manager so the caller can iterate lines
        while the connection is open.

        Separated from _post_with_retry because streaming responses cannot be
        retried transparently — partial output may already have been yielded.
        """
        response = self.session.post(
            self.url,
            json=payload,
            stream=True,
            timeout=(timeout_override if timeout_override is not None else self.timeout_s),
        )
        response.raise_for_status()
        return response

    def _post_with_retry(
        self,
        payload: dict,
        stream: bool,
        timeout_override: float | None = None,
        max_retries_override: int | None = None,
    ) -> requests.Response:
        """
        Issue a non-streaming POST with exponential-backoff retry.

        Retries only failures known to happen before request establishment.
        Once a request may be executing, retrying could duplicate generation.
        """
        actual_max_retries = max_retries_override if max_retries_override is not None else self.max_retries
        attempts = actual_max_retries + 1
        request_timeout = timeout_override if timeout_override is not None else self.timeout_s

        for attempt in range(1, attempts + 1):
            attempt_started = time.perf_counter()
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    stream=stream,
                    timeout=request_timeout,
                )
                response.raise_for_status()
                return response

            except requests.RequestException as exc:
                is_last = attempt == attempts
                if is_last or not self._is_retryable(exc):
                    category = self._inference_failure(exc).category
                    logger.warning(
                        "Ollama inference request failed category=%s attempt=%d/%d retry=false",
                        category, attempt, attempts,
                    )
                    trace_event("llm", "request_failure", payload={
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "attempt_duration_ms": round((time.perf_counter() - attempt_started) * 1000, 2),
                        "error_type": type(exc).__name__,
                        "error_category": category,
                        "retry": False,
                    })
                    raise

                backoff = self.retry_backoff_s * (2 ** (attempt - 1))
                trace_event(
                    "llm",
                    "request_retry",
                    payload={
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "attempt_duration_ms": round(
                            (time.perf_counter() - attempt_started) * 1000, 2
                        ),
                        "backoff_ms": round(backoff * 1000, 2),
                        "error_type": type(exc).__name__,
                        "status_code": getattr(
                            getattr(exc, "response", None), "status_code", None
                        ),
                    },
                )
                logger.warning(
                    "Ollama request failed (attempt %d/%d): %s — retrying in %.2fs",
                    attempt,
                    attempts,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError("unreachable")  # loop always raises or returns

    @staticmethod
    def _nanoseconds_to_ms(value):
        if not isinstance(value, (int, float)):
            return None
        return round(value / 1_000_000, 2)

    # ------------------------------------------------------------------
    # Private — utilities
    # ------------------------------------------------------------------

    def _resolve_think_value(self, think_override):
        if think_override is not None:
            return think_override
        if not self.thinking_enabled:
            return False
        return self.thinking_level or True

    def resolve_think_value(self, think_override=None):
        """Resolve a turn override against this client's configured reasoning setting."""
        return self._resolve_think_value(think_override)

    def _merge_options(self, base_options: dict, override_options: dict | None) -> dict:
        merged = dict(base_options)
        if not override_options:
            return merged
        for key, value in override_options.items():
            if value is None:
                merged.pop(key, None)
                continue
            merged[key] = value
        return merged

    def _drop_none_options(self, options: dict) -> dict:
        return {key: value for key, value in options.items() if value is not None}

    def _is_retryable(self, exc: requests.RequestException) -> bool:
        return isinstance(exc, requests.ConnectTimeout)

    @staticmethod
    def _inference_failure(exc: requests.RequestException) -> InferenceFailure:
        if isinstance(exc, requests.ConnectTimeout):
            category = "connect_timeout"
        elif isinstance(exc, requests.ReadTimeout):
            category = "read_timeout"
        elif isinstance(exc, requests.Timeout):
            category = "timeout"
        elif isinstance(exc, requests.HTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            category = "server_error" if status is not None and status >= 500 else "http_error"
        elif isinstance(exc, requests.ConnectionError):
            category = "connection_error"
        else:
            category = "request_error"
        return InferenceFailure(category, f"Ollama inference failed ({category}).")

    def _resolve_image_request_retries(
        self,
        messages,
        max_retries_override: int | None,
    ) -> int | None:
        if max_retries_override is not None:
            return max_retries_override
        if self._messages_include_images(messages):
            return 0
        return None

    # ------------------------------------------------------------------
    # Private — image fallback helpers
    # ------------------------------------------------------------------

    def _should_retry_without_images(
        self,
        exc: requests.HTTPError,
        messages,
    ) -> bool:
        if not self._multimodal_supported:
            return False

        _, has_images = self._strip_images_from_messages(messages)
        if not has_images:
            return False

        status_code = getattr(exc.response, "status_code", None)
        if status_code is None:
            return False

        if status_code not in {400, 415, 422}:
            return False

        error_text = self._http_error_text(exc)
        if not error_text:
            return True

        image_error_markers = (
            "image",
            "vision",
            "multimodal",
            "unsupported",
            "not support",
            "base64",
        )
        return any(marker in error_text for marker in image_error_markers)

    def _error_indicates_model_without_images(self, error_text: str) -> bool:
        if not error_text:
            return False

        capability_markers = (
            "does not support image",
            "doesn't support image",
            "vision is not supported",
            "multimodal is not supported",
            "model does not support vision",
        )
        return any(marker in error_text for marker in capability_markers)

    def _http_error_text(self, exc: requests.HTTPError) -> str:
        response = getattr(exc, "response", None)
        if response is None:
            return ""

        try:
            text = response.text
        except Exception:
            text = ""

        if not text:
            try:
                payload = response.json()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                text = str(payload.get("error") or payload.get("message") or "")

        return str(text).strip().lower()

    def _strip_images_from_messages(self, messages) -> tuple[list, bool]:
        stripped = []
        removed_images = False

        for message in messages:
            if not isinstance(message, dict):
                stripped.append(message)
                continue

            if "images" not in message:
                stripped.append(message)
                continue

            message_without_images = dict(message)
            message_without_images.pop("images", None)
            stripped.append(message_without_images)
            removed_images = True

        return stripped, removed_images

    def _message_images(self, message: dict) -> list:
        images = message.get("images")
        if not isinstance(images, list):
            return []
        return images

    def _messages_include_images(self, messages) -> bool:
        return any(
            isinstance(message, dict) and bool(self._message_images(message))
            for message in messages
        )

    def _image_message_indices(self, messages) -> list[int]:
        return [
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict) and self._message_images(message)
        ]

    def _current_user_message_index_with_images(self, messages) -> int | None:
        if not messages:
            return None

        index = len(messages) - 1
        message = messages[index]
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and self._message_images(message)
        ):
            return index
        return None

    def _image_count_for_message(self, messages, message_index: int | None) -> int:
        if message_index is None:
            return 0
        if message_index < 0 or message_index >= len(messages):
            return 0

        message = messages[message_index]
        images = self._message_images(message) if isinstance(message, dict) else []
        return len(images)

    def _strip_message_images(
        self,
        messages,
        message_index: int,
        image_indexes: set[int] | None,
    ) -> tuple[list, bool]:
        stripped = []
        removed_images = False

        for index, message in enumerate(messages):
            if index != message_index or not isinstance(message, dict):
                stripped.append(message)
                continue

            images = self._message_images(message)
            if not images:
                stripped.append(message)
                continue

            updated_message = dict(message)
            if image_indexes is None:
                updated_message.pop("images", None)
                removed_images = True
            else:
                remaining_images = [
                    image
                    for image_index, image in enumerate(images)
                    if image_index not in image_indexes
                ]
                if len(remaining_images) != len(images):
                    removed_images = True
                if remaining_images:
                    updated_message["images"] = remaining_images
                else:
                    updated_message.pop("images", None)

            stripped.append(updated_message)

        return stripped, removed_images

    def _build_image_fallback_messages(self, messages) -> list[tuple[list, str, int]]:
        candidates: list[tuple[list, str, int]] = []
        seen_keys: set[str] = set()

        image_message_indices = self._image_message_indices(messages)
        if not image_message_indices:
            return candidates

        current_image_message_index = self._current_user_message_index_with_images(messages)
        ordered_indices = []
        if current_image_message_index is not None:
            ordered_indices.append(current_image_message_index)
        ordered_indices.extend(
            index
            for index in reversed(image_message_indices)
            if index != current_image_message_index
        )

        for message_index in ordered_indices:
            image_count = self._image_count_for_message(messages, message_index)
            if image_count > 1:
                for image_index in range(image_count):
                    candidate, removed = self._strip_message_images(
                        messages,
                        message_index=message_index,
                        image_indexes={image_index},
                    )
                    if removed:
                        self._add_fallback_candidate(
                            candidates,
                            seen_keys,
                            candidate,
                            strategy=f"without image {image_index + 1} from message {message_index + 1}",
                            dropped_current_images_count=(
                                1 if message_index == current_image_message_index else 0
                            ),
                        )

            candidate, removed = self._strip_message_images(
                messages,
                message_index=message_index,
                image_indexes=None,
            )
            if removed:
                label = (
                    "without current message images"
                    if message_index == current_image_message_index
                    else f"without images from message {message_index + 1}"
                )
                self._add_fallback_candidate(
                    candidates,
                    seen_keys,
                    candidate,
                    strategy=label,
                    dropped_current_images_count=(
                        image_count if message_index == current_image_message_index else 0
                    ),
                )

        candidate, removed = self._strip_images_from_messages(messages)
        if removed:
            current_images_total = (
                self._image_count_for_message(messages, current_image_message_index)
                if current_image_message_index is not None
                else 0
            )
            self._add_fallback_candidate(
                candidates,
                seen_keys,
                candidate,
                strategy="without all images",
                dropped_current_images_count=current_images_total,
            )

        return candidates

    def _add_fallback_candidate(
        self,
        candidates: list[tuple[list, str, int]],
        seen_keys: set[str],
        candidate_messages: list,
        strategy: str,
        dropped_current_images_count: int,
    ) -> None:
        key = json.dumps(candidate_messages, sort_keys=True)
        if key in seen_keys:
            return
        seen_keys.add(key)
        candidates.append((candidate_messages, strategy, dropped_current_images_count))
