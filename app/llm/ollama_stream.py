import requests
import json
import time
import logging
from .base import LLMClient

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

    def chat(self, messages, think_override=None) -> str:
        """
        Non-streaming chat call.
        Used for planners, summarizers, and other structured outputs.
        """
        think_value = self._resolve_think_value(think_override)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.options,
            "think": think_value,
        }

        logger.info(
            "Ollama chat request (stream=%s, think=%r, messages=%d)",
            False,
            think_value,
            len(messages),
        )

        r = self._post_with_retry(payload, stream=False)
        r.raise_for_status()

        data = r.json()
        message = data.get("message", {})
        logger.info(
            "Ollama chat raw output: content=%r thinking=%r",
            message.get("content"),
            message.get("thinking"),
        )

        return message["content"]

    def stream_chat(self, messages, think_override=None):
        think_value = self._resolve_think_value(think_override)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options,
            "think": think_value,
        }

        logger.info(
            "Ollama chat request (stream=%s, think=%r, messages=%d)",
            True,
            think_value,
            len(messages),
        )

        collected_content = []
        collected_thinking = []

        with self._post_with_retry(payload, stream=True) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line.decode("utf-8"))
                message = chunk.get("message", {})
                content = message.get("content")
                thinking = message.get("thinking")

                if content:
                    collected_content.append(content)
                    yield content

                if thinking:
                    collected_thinking.append(thinking)

                if chunk.get("done"):
                    break

        logger.info(
            "Ollama stream raw output: content=%r thinking=%r",
            "".join(collected_content),
            "".join(collected_thinking) or None,
        )

    def _post_with_retry(self, payload: dict, stream: bool):
        attempts = self.max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    stream=stream,
                    timeout=self.timeout_s,
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
