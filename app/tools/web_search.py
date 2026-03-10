import requests
import logging
import time

logger = logging.getLogger(__name__)


class WebSearchResult:
    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content


class SearXNGClient:
    """
    Low-level HTTP client for SearXNG.
    Responsible only for talking to the service.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 10.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.25,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.is_available: bool = False

    def probe(self) -> bool:
        """
        Check whether the SearXNG instance is reachable.
        This should be called once at startup.
        """
        try:
            requests.get(
                f"{self.base_url}/search",
                params={"q": "ping", "format": "json"},
                timeout=2.0,
            )
            self.is_available = True
        except Exception:
            self.is_available = False

        return self.is_available

    def search(self, query: str, limit: int = 5) -> list[WebSearchResult]:
        params = {
            "q": query,
            "format": "json",
        }

        response = self._get_with_retry(
            f"{self.base_url}/search",
            params=params,
        )
        response.raise_for_status()

        data = response.json()
        results = []

        for r in data.get("results", [])[:limit]:
            results.append(
                WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                )
            )

        return results

    def _get_with_retry(self, url: str, params: dict):
        attempts = self.max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout,
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
                    "Web search request failed (attempt %d/%d): %s. Retrying in %.2fs",
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


class WebSearchTool:
    """
    Orchestrator-facing tool adapter.
    Wraps the client and summarizer into a single optional tool.
    """

    name = "web_search"

    def __init__(self, client: SearXNGClient, summarizer):
        self.client = client
        self.summarizer = summarizer

    @property
    def is_available(self) -> bool:
        return self.client.is_available

    def run(self, query: str) -> str | None:
        """
        Execute the web search and return a summarized context string.
        Raises on unexpected failure (orchestrator handles it).
        """
        results = self.client.search(query)
        summary = self.summarizer.summarize(results)

        if not summary:
            return None

        return f"External information:\n{summary}"
