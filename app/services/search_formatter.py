def format_search_results(results) -> str:
    lines = ["Web search results:"]

    for r in results:
        snippet = r.content.strip()
        if snippet:
            snippet = snippet.replace("\n", " ")
        lines.append(f"- {r.title}: {snippet}")

    return "\n".join(lines)
