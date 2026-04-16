import requests
import json
import time
import logging
from typing import Iterator

from .base import LLMClient
from app.logging import trace_event

logger = logging.getLogger("ollama_client")

# Options that are OpenAI-specific and not understood by Ollama.
_OPENAI_ONLY_OPTIONS = frozenset({"frequency_penalty", "presence_penalty"})

# Stop sequences appended unconditionally to every streaming request to
# prevent the model from rambling past a natural end-of-turn token.
_BUILTIN_STOP_SEQUENCES = ["<|eot_id|>", "<|im_end|>", "<|end_of_sentence|>"]

# Default repetition penalty applied when the caller has not set one.
_DEFAULT_REPEAT_PENALTY = 1.15

# Maximum temperature allowed for streaming responses. If the configured
# value exceeds this cap it is clamped down, not replaced with a hardcoded 1.
_MAX_STREAM_TEMPERATURE = 1.5

# Minimum context / prediction window sizes used when thinking is active.
_THINKING_MIN_CTX = 65_536
_THINKING_MIN_PREDICT = 32_768


class OllamaClient(LLMClient):
    """
    Ollama-backed LLM client.

    Supports both blocking (chat) and streaming (stream_chat) call patterns.
    Thinking tokens (extended reasoning) are an opt-in feature controlled by
    `thinking_enabled` and `thinking_level`.

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
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

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
    ) -> str:
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

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": request_options,
            "think": think_value,
        }

        logger.info(
            "Ollama chat request (stream=False, think=%r, messages=%d)",
            think_value,
            len(messages),
        )
        trace_event("llm", "chat_request", payload=payload)

        # _post_with_retry already calls raise_for_status() internally.
        r = self._post_with_retry(
            payload,
            stream=False,
            timeout_override=timeout_override,
            max_retries_override=max_retries_override,
        )

        data = r.json()
        message = data.get("message", {})
        trace_event(
            "llm",
            "chat_response",
            payload={
                "content": message.get("content"),
                "thinking": message.get("thinking"),
                "done_reason": data.get("done_reason"),
            },
        )

        return message["content"]

    def stream_chat(self, messages, think_override=None) -> Iterator[str]:
        """
        Streaming call. Yields text chunks for user-facing responses.

        Thinking tokens are wrapped in <think>…</think> and yielded inline
        so the orchestrator can process or strip them downstream.
        """
        think_value = self._resolve_think_value(think_override)
        request_options = self._build_stream_options(think_value)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": request_options,
            "think": think_value,
        }

        logger.info(
            "Ollama chat request (stream=True, think=%r, messages=%d)",
            think_value,
            len(messages),
        )
        trace_event("llm", "stream_request", payload=payload)

        collected_content: list[str] = []
        collected_thinking: list[str] = []
        in_thinking_block = False

        with self._post_stream(payload) as r:
            for line in r.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line.decode("utf-8"))
                message = chunk.get("message", {})
                content = message.get("content")
                thinking = message.get("thinking")

                # 1. Handle thinking tokens.
                if thinking:
                    if not in_thinking_block:
                        yield "<think>\n"
                        in_thinking_block = True
                    collected_thinking.append(thinking)
                    yield thinking

                # 2. Close the thinking block when content starts arriving.
                if content and in_thinking_block:
                    yield "\n</think>\n\n"
                    in_thinking_block = False

                # 3. Handle content tokens.
                if content:
                    collected_content.append(content)
                    yield content

                # 4. End of stream.
                if chunk.get("done"):
                    if in_thinking_block:
                        yield "\n</think>\n\n"
                    if chunk.get("done_reason") == "length":
                        logger.warning(
                            "Ollama stream hit the token limit (num_predict) "
                            "before finishing — response may be truncated."
                        )
                    break

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
    # Private — request helpers
    # ------------------------------------------------------------------

    def _build_stream_options(self, think_value) -> dict:
        """
        Build the Ollama options dict for a streaming request.

        Applies several safety defaults on top of the instance options:
        - Renames max_tokens → num_predict (Ollama's key).
        - Drops OpenAI-only keys Ollama does not understand.
        - Clamps temperature to _MAX_STREAM_TEMPERATURE if it exceeds it.
        - Sets a default repeat_penalty if none is configured.
        - Appends built-in stop sequences.
        - Boosts context / prediction limits when thinking is active.
        """
        opts = self.options.copy()

        # Rename OpenAI key to Ollama equivalent.
        if "max_tokens" in opts:
            opts["num_predict"] = opts.pop("max_tokens")

        # Drop keys Ollama does not recognise.
        for key in _OPENAI_ONLY_OPTIONS:
            opts.pop(key, None)

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
            opts["num_predict"] = max(opts.get("num_predict", _THINKING_MIN_PREDICT), _THINKING_MIN_PREDICT)

        return opts

    def _post_stream(self, payload: dict) -> requests.Response:
        """
        Issue a streaming POST. Returns the raw Response used as a context
        manager so the caller can iterate lines while the connection is open.

        Separated from _post_with_retry because streaming responses cannot be
        retried transparently — partial output may already have been yielded.
        """
        request_timeout = self.timeout_s
        response = self.session.post(
            self.url,
            json=payload,
            stream=True,
            timeout=request_timeout,
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

        Retries on transient network errors and 5xx responses.
        Raises immediately on 4xx or non-retryable errors.
        """
        actual_max_retries = max_retries_override if max_retries_override is not None else self.max_retries
        attempts = actual_max_retries + 1
        request_timeout = timeout_override if timeout_override is not None else self.timeout_s

        for attempt in range(1, attempts + 1):
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
                    raise

                backoff = self.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Ollama request failed (attempt %d/%d): %s — retrying in %.2fs",
                    attempt,
                    attempts,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError("unreachable")  # loop always raises or returns

    # ------------------------------------------------------------------
    # Private — utilities
    # ------------------------------------------------------------------

    def _resolve_think_value(self, think_override):
        if think_override is not None:
            return think_override
        if not self.thinking_enabled:
            return False
        return self.thinking_level or True

    def _is_retryable(self, exc: requests.RequestException) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError):
            status_code = getattr(exc.response, "status_code", None)
            return status_code is not None and status_code >= 500
        return False
