import requests


class WebSearchResult:
    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content


class SearXNGClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, limit: int = 5) -> list[WebSearchResult]:
        params = {
            "q": query,
            "format": "json",
        }

        response = requests.get(
            f"{self.base_url}/search",
            params=params,
            timeout=10,
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
