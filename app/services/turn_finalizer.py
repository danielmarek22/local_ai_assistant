import logging

from app.logging import trace_event

logger = logging.getLogger("turn_finalizer")


class TurnFinalizer:
    def __init__(
        self,
        history_store,
        summary_store,
        summarizer,
        summary_trigger: int = 10,
        completion_observers=None,
    ):
        self.history = history_store
        self.summary_store = summary_store
        self.summarizer = summarizer
        self.summary_trigger = summary_trigger
        self.completion_observers = list(completion_observers or [])

    def finalize(self, session_id: str, completed_turn=None) -> None:
        if completed_turn is not None:
            for observer in self.completion_observers:
                try:
                    observer.observe(completed_turn)
                except Exception:
                    logger.exception(
                        "[%s] Turn completion observer failed; continuing finalization",
                        session_id,
                    )

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
        trace_event(
            "turn_finalizer",
            "summarization_check",
            session_id=session_id,
            payload={
                "current_count": current_count,
                "last_count": last_count,
                "summary_trigger": self.summary_trigger,
            },
        )

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
        trace_event(
            "turn_finalizer",
            "summary_input",
            session_id=session_id,
            payload={"summary_input": summary_input},
        )

        try:
            summary = self.summarizer.summarize(summary_input)
        except Exception:
            logger.exception("[%s] Summarization failed", session_id)
            return

        self.summary_store.set(session_id, summary, current_count)
        logger.info("[%s] History summarized (%d chars)", session_id, len(summary))
        trace_event(
            "turn_finalizer",
            "summary_saved",
            session_id=session_id,
            payload={"summary": summary, "last_turn_count": current_count},
        )
