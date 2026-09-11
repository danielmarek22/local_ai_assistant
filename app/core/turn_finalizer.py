import logging

from app.logging import trace_event
from app.core.conversation import SessionKind, render_group_message

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
        if summary_trigger < 1:
            raise ValueError("summary_trigger must be at least one")
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
            existing_summary, last_message_id = summary_data
        else:
            existing_summary, last_message_id = None, 0

        # One bounded batch per completed turn; preserve backlog for later turns.
        history = self.history.get_summary_batch(
            session_id=session_id,
            after_message_id=last_message_id,
            limit=max(100, self.summary_trigger),
        )
        trace_event(
            "turn_finalizer",
            "summarization_check",
            session_id=session_id,
            payload={
                "pending_batch_count": len(history),
                "last_message_id": last_message_id,
                "summary_trigger": self.summary_trigger,
            },
        )

        if len(history) < self.summary_trigger:
            return

        logger.info("[%s] Summarizing conversation history", session_id)

        summary_input = [
            {
                "role": row["role"],
                "content": self._summary_content(session_id, row),
            }
            for row in history
        ]
        trace_event(
            "turn_finalizer",
            "summary_input",
            session_id=session_id,
            payload={"previous_summary": existing_summary, "summary_input": summary_input},
        )

        try:
            summary = self.summarizer.summarize(
                summary_input,
                previous_summary=existing_summary,
            )
        except Exception:
            logger.exception("[%s] Summarization failed", session_id)
            return

        self.summary_store.set(session_id, summary, history[-1]["id"])
        logger.info("[%s] History summarized (%d chars)", session_id, len(summary))
        trace_event(
            "turn_finalizer",
            "summary_saved",
            session_id=session_id,
            payload={"summary": summary, "last_message_id": history[-1]["id"]},
        )

    def _summary_content(self, session_id: str, row: dict) -> str:
        get_session_kind = getattr(self.history, "get_session_kind", None)
        if not callable(get_session_kind) or get_session_kind(session_id) == SessionKind.DIRECT:
            return row["content"]
        effective_sender = getattr(self.history, "effective_sender", None)
        if not callable(effective_sender):
            return row["content"]
        return render_group_message(row["content"], effective_sender(row))
