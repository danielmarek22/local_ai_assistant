import logging


logger = logging.getLogger("turn_finalizer")


class TurnFinalizer:
    def __init__(
        self,
        history_store,
        summary_store,
        summarizer,
        summary_trigger: int = 10,
    ):
        self.history = history_store
        self.summary_store = summary_store
        self.summarizer = summarizer
        self.summary_trigger = summary_trigger

    def finalize(self, session_id: str) -> None:
        logger.debug("[%s] Checking summarization conditions", session_id)

        summary_data = self.summary_store.get(session_id)
        if summary_data:
            existing_summary, last_count = summary_data
        else:
            existing_summary, last_count = None, 0

        history = self.history.get_recent(
            session_id=session_id,
            limit=1000,
        )
        current_count = len(history)

        if (current_count - last_count) < self.summary_trigger:
            return

        logger.info("[%s] Summarizing conversation history", session_id)

        summary_input = []
        if existing_summary:
            summary_input.append(
                {
                    "role": "system",
                    "content": (
                        "Here is the current summary of the conversation so far. "
                        "Update it using the new messages below:\n\n"
                        f"{existing_summary}"
                    ),
                }
            )

        summary_input.extend(
            {"role": row["role"], "content": row["content"]}
            for row in history[last_count:]
        )

        try:
            summary = self.summarizer.summarize(summary_input)
        except Exception:
            logger.exception("[%s] Summarization failed", session_id)
            return

        self.summary_store.set(session_id, summary, current_count)
        logger.info("[%s] History summarized (%d chars)", session_id, len(summary))
