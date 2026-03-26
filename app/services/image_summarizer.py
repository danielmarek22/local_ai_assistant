class ImageSummarizer:
    def __init__(self, llm):
        self.llm = llm

    def summarize(self, attachment, message_text: str = "") -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are generating a factual summary of a user-provided image for long-term memory retrieval.\n\n"
                    "Rules:\n"
                    "- Describe only what is visibly present in the image.\n"
                    "- Include visible text, labels, UI states, warnings, or error messages when relevant.\n"
                    "- Do not speculate about intent, identity, or unseen context.\n"
                    "- Keep the result concise and retrieval-friendly.\n"
                    "- Use plain English in 1 to 3 sentences.\n"
                    "- Output only the summary text."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize this image for future retrieval.\n"
                    f"Attachment name: {attachment.name}\n"
                    f"Related message text: {message_text or '(none)'}"
                ),
                "images": [attachment.to_llm_image()],
            },
        ]

        return self.llm.chat(
            prompt,
            think_override=False,
            options_override={
                "temperature": 0.1,
                "num_predict": 160,
            },
        ).strip()
