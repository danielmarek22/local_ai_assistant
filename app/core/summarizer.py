class HistorySummarizer:
    def __init__(self, llm):
        self.llm = llm

    def summarize(self, messages: list[dict]) -> str:
        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation briefly. "
                    "Focus on facts, decisions, and user preferences. "
                    "Do not include dialogue or filler."
                )
            }
        ]

        summary_prompt.extend(messages)

        buffer = ""
        for chunk in self.llm.stream_chat(summary_prompt):
            buffer += chunk

        return buffer.strip()
