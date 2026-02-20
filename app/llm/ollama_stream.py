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
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.25,
    ):
        self.model = model
        self.url = f"{host}/v1/chat/completions"
        self.options = options or {}
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def chat(self, messages) -> str:
        """
        Non-streaming chat call.
        Used for planners, summarizers, and other structured outputs.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.options,
        }

        r = self._post_with_retry(payload, stream=False)
        r.raise_for_status()

        data = r.json()

        return data["choices"][0]["message"]["content"]

    def stream_chat(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options,
        }

        with self._post_with_retry(payload, stream=True) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")

                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data: ").strip()

                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    yield delta["content"]

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

    def _is_retryable(self, exc: requests.RequestException) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True

        if isinstance(exc, requests.HTTPError):
            status_code = getattr(exc.response, "status_code", None)
            return status_code is not None and status_code >= 500

        return False
