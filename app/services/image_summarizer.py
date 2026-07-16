import logging

from app.logging import trace_event

logger = logging.getLogger("image_summarizer")

class ImageSummarizer:
    def __init__(self, llm):
        self.llm = llm

    def summarize(self, attachment, message_text: str = "") -> str | None:
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

        response = self.llm.chat(
            prompt,
            think_override=False,
            options_override={
                "temperature": 0.1,
                "num_predict": 160,
            },
        )
        
        # Safely extract the text content from the dictionary before stripping
        result = response.get("content", "").strip()
        
        if getattr(self.llm, "last_chat_dropped_current_images", False):
            logger.warning(
                "Skipping image summary for %r because Ollama rejected the image payload.",
                attachment.name,
            )
            return None

        trace_event(
            "image_summarizer",
            "summary_generated",
            payload={
                "attachment": {
                    "name": attachment.name,
                    "mime_type": attachment.mime_type,
                    "size_bytes": attachment.size_bytes,
                },
                "prompt": prompt,
                "summary": result,
            },
        )
        return result
