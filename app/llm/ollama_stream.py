import requests
import json
import time
import logging
from typing import Iterator

from .base import LLMClient
from app.core.thinking_filter import ThinkingBlockSplitter
from app.logging import trace_event

logger = logging.getLogger("ollama_client")

# Stop sequences appended unconditionally to every streaming request to
# prevent the model from rambling past a natural end-of-turn token.
_BUILTIN_STOP_SEQUENCES = ["<|eot_id|>", "<|im_end|>", "<|end_of_sentence|>"]

# Default repetition penalty applied when the caller has not set one.
_DEFAULT_REPEAT_PENALTY = 1.15

# Maximum temperature allowed for streaming responses. If the configured
# value exceeds this cap it is clamped down, not replaced with a hardcoded 1.
_MAX_STREAM_TEMPERATURE = 1.5

# Minimum context / prediction window sizes used when thinking is active.
_THINKING_MIN_CTX = 8192
_THINKING_MIN_PREDICT = 2048


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
        thinking_options: dict | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.25,
    ):
        self.model = model
        self._preload_url = f"{host}/api/generate"
        self.url = f"{host}/v1/chat/completions"
        self.options = options or {}
        self.thinking_enabled = thinking_enabled
        self.thinking_level = thinking_level
        self.thinking_options = thinking_options or {}
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._multimodal_supported = True
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

        payload = self._build_payload(
            messages,
            stream=False,
            think_value=think_value,
            options=request_options,
        )

        logger.info(
            "Ollama chat request (stream=False, reasoning_effort=%r, messages=%d)",
            payload.get("reasoning_effort"),
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
        choice = self._first_choice(data)
        message = choice.get("message", {})
        trace_event(
            "llm",
            "chat_response",
            payload={
                "content": message.get("content"),
                "thinking": message.get("reasoning"),
                "done_reason": choice.get("finish_reason"),
            },
        )

        return message.get("content", "")

    def stream_chat(self, messages, think_override=None) -> Iterator[str]:
        """
        Streaming call. Yields text chunks for user-facing responses.

        Thinking tokens are wrapped in <think>…</think> and yielded inline
        so the orchestrator can process or strip them downstream.
        """
        think_value = self._resolve_think_value(think_override)
        request_options = self._build_stream_options(think_value)
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
        )

        logger.info(
            "Ollama chat request (stream=True, reasoning_effort=%r, messages=%d)",
            payload.get("reasoning_effort"),
            len(messages),
        )
        trace_event("llm", "stream_request", payload=payload)

        collected_content: list[str] = []
        collected_thinking: list[str] = []
        try:
            yield from self._stream_payload(payload, collected_content, collected_thinking)
        except requests.HTTPError as exc:
            if not self._should_retry_stream_without_images(exc, request_messages):
                raise

            response_text = self._http_error_text(exc)
            if self._error_indicates_model_without_images(response_text):
                self._multimodal_supported = False

            fallback_candidates = self._build_image_fallback_messages(request_messages)
            if not fallback_candidates:
                raise

            last_exc: requests.HTTPError = exc
            for fallback_messages, strategy, dropped_current_images_count in fallback_candidates:
                payload["messages"] = self._prepare_messages(fallback_messages)
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
                    yield from self._stream_payload(payload, collected_content, collected_thinking)
                    self.last_stream_image_fallback_strategy = strategy
                    if dropped_current_images_count > 0:
                        self.last_stream_dropped_current_images = True
                        self.last_stream_dropped_current_images_count = dropped_current_images_count
                    break
                except requests.HTTPError as retry_exc:
                    if not self._should_retry_stream_without_images(retry_exc, fallback_messages):
                        raise
                    last_exc = retry_exc
                    if self._error_indicates_model_without_images(self._http_error_text(retry_exc)):
                        self._multimodal_supported = False
            else:
                raise last_exc

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

    def _stream_payload(
        self,
        payload: dict,
        collected_content: list[str],
        collected_thinking: list[str],
    ) -> Iterator[str]:
        in_thinking_block = False
        inline_thinking_splitter = ThinkingBlockSplitter()

        with self._post_stream(payload) as r:
            for line in r.iter_lines():
                if not line:
                    continue

                decoded_line = line.decode("utf-8").strip()
                if not decoded_line or decoded_line.startswith(":"):
                    continue
                if not decoded_line.startswith("data:"):
                    continue

                data_payload = decoded_line[5:].strip()
                if not data_payload:
                    continue
                if data_payload == "[DONE]":
                    self._flush_stream_tail(
                        inline_thinking_splitter,
                        collected_content,
                        collected_thinking,
                        in_thinking_block,
                    )
                    break

                chunk = json.loads(data_payload)
                choice = self._first_choice(chunk)
                delta = choice.get("delta", {})
                content = delta.get("content")
                thinking = delta.get("reasoning") or delta.get("reasoning_content")
                inline_visible = ""
                inline_thinking = ""

                if content:
                    inline_visible, inline_thinking = inline_thinking_splitter.push(content)

                combined_thinking = "".join(
                    part for part in (thinking, inline_thinking) if part
                )

                # 1. Handle thinking tokens.
                if combined_thinking:
                    if not in_thinking_block:
                        yield "<think>\n"
                        in_thinking_block = True
                    collected_thinking.append(combined_thinking)
                    yield combined_thinking

                # 2. Close the thinking block when content starts arriving.
                if inline_visible and in_thinking_block:
                    yield "\n</think>\n\n"
                    in_thinking_block = False

                # 3. Handle content tokens.
                if inline_visible:
                    collected_content.append(inline_visible)
                    yield inline_visible

                # 4. End of stream.
                if choice.get("finish_reason") is not None:
                    yield from self._flush_stream_tail(
                        inline_thinking_splitter,
                        collected_content,
                        collected_thinking,
                        in_thinking_block,
                    )
                    if choice.get("finish_reason") == "length":
                        logger.warning(
                            "Ollama stream hit the token limit (max_tokens) "
                            "before finishing — response may be truncated."
                        )
                    break

    # ------------------------------------------------------------------
    # Private — request helpers
    # ------------------------------------------------------------------

    def _build_stream_options(self, think_value) -> dict:
        """
        Build the chat-completions request fields for a streaming request.

        Applies several safety defaults on top of the instance options:
        - Applies thinking-specific overrides when thinking is active.
        - Renames num_predict → max_tokens for the OpenAI-compatible API.
        - Clamps temperature to _MAX_STREAM_TEMPERATURE if it exceeds it.
        - Sets a default repeat_penalty if none is configured.
        - Appends built-in stop sequences.
        - Boosts context / prediction limits when thinking is active.
        """
        opts = self.options.copy()
        if think_value:
            opts = self._merge_options(opts, self.thinking_options)

        opts = self._drop_none_options(opts)

        if "num_predict" in opts and "max_tokens" not in opts:
            opts["max_tokens"] = opts.pop("num_predict")

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
            opts["max_tokens"] = max(opts.get("max_tokens", _THINKING_MIN_PREDICT), _THINKING_MIN_PREDICT)

        return opts

    def _build_payload(
        self,
        messages,
        stream: bool,
        think_value,
        options: dict | None = None,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            "stream": stream,
            "reasoning_effort": self._reasoning_effort(think_value),
        }
        if options:
            payload.update(self._normalize_request_options(options))
        return payload

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
        return {
            key: value
            for key, value in options.items()
            if value is not None
        }

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

    def _reasoning_effort(self, think_value) -> str:
        if not think_value:
            return "none"
        if isinstance(think_value, str):
            return think_value
        return "medium"

    def _normalize_request_options(self, options: dict) -> dict:
        normalized = self._drop_none_options(options)
        if "num_predict" in normalized and "max_tokens" not in normalized:
            normalized["max_tokens"] = normalized.pop("num_predict")
        return normalized

    def _prepare_messages(self, messages) -> list:
        prepared = []

        for message in messages:
            if not isinstance(message, dict):
                prepared.append(message)
                continue

            updated_message = dict(message)
            images = self._message_images(updated_message)
            if images:
                updated_message["content"] = self._build_multimodal_content(
                    updated_message.get("content"),
                    images,
                )
                updated_message.pop("images", None)
            prepared.append(updated_message)

        return prepared

    def _build_multimodal_content(self, content, images: list) -> list[dict]:
        parts: list[dict] = []

        if isinstance(content, list):
            parts.extend(content)
        elif content not in (None, ""):
            parts.append({"type": "text", "text": str(content)})

        for image in images:
            if not isinstance(image, str) or not image:
                continue
            image_url = image if image.startswith("data:") else f"data:image/png;base64,{image}"
            parts.append({"type": "image_url", "image_url": image_url})

        return parts

    def _first_choice(self, payload: dict) -> dict:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                return first_choice
        return {}

    def _flush_stream_tail(
        self,
        inline_thinking_splitter: ThinkingBlockSplitter,
        collected_content: list[str],
        collected_thinking: list[str],
        in_thinking_block: bool,
    ) -> Iterator[str]:
        final_visible, final_thinking = inline_thinking_splitter.flush()
        if final_thinking:
            if not in_thinking_block:
                yield "<think>\n"
                in_thinking_block = True
            collected_thinking.append(final_thinking)
            yield final_thinking
        if final_visible and in_thinking_block:
            yield "\n</think>\n\n"
            in_thinking_block = False
        if final_visible:
            collected_content.append(final_visible)
            yield final_visible
        if in_thinking_block:
            yield "\n</think>\n\n"

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

    def _message_images(self, message: dict) -> list:
        images = message.get("images")
        if not isinstance(images, list):
            return []
        return images

    def _should_retry_stream_without_images(
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

        # Some Ollama backends report malformed/unsupported image payloads
        # as 5xx instead of 4xx. If the request includes images, allow the
        # image-stripping fallback path to try recovering.
        if status_code >= 500:
            return True

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

    def _is_retryable(self, exc: requests.RequestException) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError):
            status_code = getattr(exc.response, "status_code", None)
            return status_code is not None and status_code >= 500
        return False
