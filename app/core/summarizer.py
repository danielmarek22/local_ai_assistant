class HistorySummarizer:
    def __init__(self, llm):
        self.llm = llm

    def summarize(self, messages: list[dict]) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are generating a factual summary of a conversation.\n\n"
                    "Rules:\n"
                    "- Summarize ONLY information that was explicitly stated.\n"
                    "- Do NOT add new facts, advice, ideas, or interpretations.\n"
                    "- Do NOT infer intentions or preferences unless explicitly stated.\n"
                    "- Do NOT include opinions, tone, or conversational filler.\n"
                    "- Do NOT include instructions, recipes, recommendations, or steps.\n"
                    "- Do NOT include internal reasoning, explanations, or meta-commentary.\n"
                    "- Do NOT include markup, tags, or special tokens.\n"
                    "- Use plain, neutral English.\n"
                    "- Write in complete sentences.\n"
                    "- Keep the summary concise (3–6 sentences).\n"
                    "- The summary must remain correct even if read out of context.\n\n"
                    "Output only the summary text."
                )
            }
        ]

        # Only include user + assistant messages
        for m in messages:
            if m["role"] in ("user", "assistant"):
                prompt.append(m)

        buffer = ""
        for chunk in self.llm.stream_chat(prompt):
            buffer += chunk

        return buffer.strip()
