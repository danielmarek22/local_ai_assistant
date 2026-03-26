class SearchResultSummarizer:
    def __init__(self, llm):
        self.llm = llm

    def summarize(self, results: list) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are summarizing web search results.\n\n"
                    "Rules:\n"
                    "- Use ONLY the information present in the provided search results.\n"
                    "- Do NOT add new facts, quantities, ratios, or steps.\n"
                    "- Do NOT infer missing details.\n"
                    "- Do NOT explain, justify, or comment.\n"
                    "- Do NOT mention sources, URLs, or the word \"search\".\n"
                    "- Do NOT include opinions, advice, or instructions.\n"
                    "- Use neutral, factual language.\n"
                    "- Write in short sentences.\n"
                    "- The summary must remain correct if read in isolation.\n\n"
                    "Output only the summary text."
                )
            }
        ]

        # Format and concatenate all result snippets into one user message
        formatted_snippets = [
            f"--- Result {i+1} ---\n{r.content.strip()}" 
            for i, r in enumerate(results) if r.content.strip()
        ]
        
        if formatted_snippets:
            combined_content = "Search Results:\n\n" + "\n\n".join(formatted_snippets)
            prompt.append({
                "role": "user",
                "content": combined_content
            })

        buffer = ""
        for chunk in self.llm.stream_chat(prompt):
            buffer += chunk

        return buffer.strip()