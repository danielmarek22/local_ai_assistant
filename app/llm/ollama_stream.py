import requests
import json
from typing import Iterator
from .base import LLMClient


class OllamaClient(LLMClient):
    def __init__(self, model: str, host: str, options: dict | None = None):
        self.model = model
        self.url = f"{host}/v1/chat/completions"
        self.options = options or {}

    def stream_chat(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options
        }

        with requests.post(self.url, json=payload, stream=True) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")

                # Ollama uses SSE-style streaming
                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data: ").strip()

                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]

                if "content" in delta:
                    yield delta["content"]
