import requests
import json
import time
import logging
from .base import LLMClient
from app.logging import trace_event

logger = logging.getLogger("ollama_client")


class OllamaClient(LLMClient):
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
        self.url = f"{host}/api/chat"
        self.options = options or {}
        self.thinking_enabled = thinking_enabled
        self.thinking_level = thinking_level
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        
        # Use a Session to reuse the TCP connection for faster, lower-latency requests
        self.session = requests.Session()

    def chat(
        self,
        messages,
        think_override=None,
        options_override: dict | None = None,
        timeout_override: float | None = None,
        max_retries_override: int | None = None,  # <-- Added parameter
        ) -> str:
        """
        Non-streaming chat call.
        Used for planners, summarizers, and other structured outputs.
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
            "Ollama chat request (stream=%s, think=%r, messages=%d)",
            False,
            think_value,
            len(messages),
        )
        trace_event("llm", "chat_request", payload=payload)

        # Pass both overrides down to the post method
        r = self._post_with_retry(
            payload, 
            stream=False, 
            timeout_override=timeout_override,
            max_retries_override=max_retries_override  # <-- Pass it down
        )
        r.raise_for_status()

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

    def stream_chat(self, messages, think_override=None):
        think_value = self._resolve_think_value(think_override)
        
        # 1. Create a shallow copy to avoid mutating the class-level options
        request_options = self.options.copy()
            
        # --- Map max_tokens from your YAML to Ollama's num_predict ---
        if "max_tokens" in request_options:
            request_options["num_predict"] = request_options.pop("max_tokens")

        # --- Anti-Rambling Failsafes ---
        # Clean up OpenAI specific params so Ollama doesn't complain
        request_options.pop("frequency_penalty", None)
        request_options.pop("presence_penalty", None) 
        
        # Force a strict repetition penalty to break infinite loops.
        if "repeat_penalty" not in request_options:
            request_options["repeat_penalty"] = 1.15

        # Lower the temperature
        if "temperature" not in request_options or request_options["temperature"] > 0.6:
            request_options["temperature"] = 0.6
        
        # Ensure stop sequences exist
        existing_stops = request_options.get("stop", [])
        if isinstance(existing_stops, str):
            existing_stops = [existing_stops]
        request_options["stop"] = existing_stops + ["<|eot_id|>", "<|im_end|>", "<|end_of_sentence|>"]

        # 2. Dynamically boost the token limit if thinking is enabled
        if think_value:
            request_options["num_ctx"] = max(request_options.get("num_ctx", 65536), 65536)
            request_options["num_predict"] = max(request_options.get("num_predict", 32768), 32768)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": request_options,
            "think": think_value,
        }

        logger.info(
            "Ollama chat request (stream=%s, think=%r, messages=%d)",
            True,
            think_value,
            len(messages),
        )
        trace_event("llm", "stream_request", payload=payload)

        collected_content = []
        collected_thinking = []
        in_thinking_block = False

        with self._post_with_retry(payload, stream=True) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue

                # Kept the decode step just to be safe with older JSON parsers
                chunk = json.loads(line.decode("utf-8"))
                message = chunk.get("message", {})
                content = message.get("content")
                thinking = message.get("thinking")

                # 1. Handle thinking tokens
                if thinking:
                    if not in_thinking_block:
                        yield "<think>\n"
                        in_thinking_block = True
                    
                    collected_thinking.append(thinking)
                    yield thinking

                # 2. Close the thinking block if content starts arriving
                if content and in_thinking_block:
                    yield "\n</think>\n\n"
                    in_thinking_block = False

                # 3. Handle actual content tokens
                if content:
                    collected_content.append(content)
                    yield content

                # 4. Handle the end of the stream
                if chunk.get("done"):
                    # Catch unclosed tags if the model abruptly stops
                    if in_thinking_block:
                        yield "\n</think>\n\n"
                    
                    # Log if we ran out of tokens during generation
                    if chunk.get("done_reason") == "length":
                        logger.warning(
                            "Ollama stream hit the token limit (num_predict) before finishing. "
                            "Content may be missing!"
                        )
                    break

        logger.info(
            "Ollama stream raw output: content_len=%d thinking_len=%d",
            len("".join(collected_content)),
            len("".join(collected_thinking))
        )
        trace_event(
            "llm",
            "stream_response",
            payload={
                "content": "".join(collected_content),
                "thinking": "".join(collected_thinking),
            },
        )

    def _post_with_retry(
        self, 
        payload: dict, 
        stream: bool, 
        timeout_override: float | None = None,
        max_retries_override: int | None = None,
    ):
        # Determine which retry count to use
        actual_max_retries = max_retries_override if max_retries_override is not None else self.max_retries
        attempts = actual_max_retries + 1
        
        # Determine which timeout to use
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
                retryable = self._is_retryable(exc)
                is_last = attempt == attempts

                if is_last or not retryable:
                    raise

                backoff = self.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Ollama request failed (attempt %d/%d): %s. Retrying in %.2fs",
                    attempt,
                    attempts,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError("unreachable")

    def _resolve_think_value(self, think_override):
        if think_override is not None:
            return think_override

        if not self.thinking_enabled:
            return False

        if self.thinking_level:
            return self.thinking_level

        return True

    def _is_retryable(self, exc: requests.RequestException) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True

        if isinstance(exc, requests.HTTPError):
            status_code = getattr(exc.response, "status_code", None)
            return status_code is not None and status_code >= 500

        return False
