import json
import logging
import time

from app.logging import trace_event

logger = logging.getLogger("scratchpad_context")


def bounded_entries(entries, max_chars, *, preview_chars=500):
    """Keep whole JSON records and explicit previews within the output budget."""
    records = []
    for entry in entries:
        content = entry["content"]
        record = {**entry, "content": content[:preview_chars],
                  "truncated": len(content) > preview_chars}
        # Escaping keeps embedded newlines/section markers inside a JSON string.
        while True:
            candidate = json.dumps([*records, record], ensure_ascii=False, separators=(",", ":"))
            if len(candidate) <= max_chars:
                records.append(record)
                break
            if not record["content"]:
                break
            excess = len(candidate) - max_chars
            record["content"] = record["content"][:max(0, len(record["content"]) - max(1, excess))]
            record["truncated"] = True
        if not records or records[-1] is not record:
            break
    return records


class ScratchpadContextProvider:
    def __init__(self, store, history_store, *, recall_limit=3, query_max_chars=2000,
                 context_max_chars=3000):
        self.store = store
        self.history = history_store
        self.recall_limit = recall_limit
        self.query_max_chars = query_max_chars
        self.context_max_chars = context_max_chars

    def query_for_turn(self, session_id, current_text, current_message_id=None):
        current = current_text.strip()[:self.query_max_chars]
        if not current:
            return ""
        # Fetch a few extra rows for non-conversational/error records and the current row.
        rows = self.history.get_recent(session_id, limit=8, conversation_only=True)
        preceding = [row for row in rows
                     if row.get("role") in {"user", "assistant"}
                     and not row.get("retry_error") and not row.get("excluded_from_context")
                     and (current_message_id is None or row["id"] < current_message_id)][-2:]
        remaining = self.query_max_chars - len(current)
        parts = []
        for row in reversed(preceding):
            text = str(row.get("content", "")).strip()
            if not text or remaining <= 1:
                continue
            label = row.get("sender_display_name") or row["role"]
            part = f"{label}: {text}"[:remaining - 1]
            parts.insert(0, part)
            remaining -= len(part) + 1
        return "\n".join([*parts, current])

    def context_for_turn(self, session_id, current_text, current_message_id=None):
        started = time.perf_counter()
        failed = False
        entries = []
        try:
            query = self.query_for_turn(session_id, current_text, current_message_id)
            if query:
                entries = self.store.search(query, limit=self.recall_limit)
            records = bounded_entries(entries, self.context_max_chars, preview_chars=1000)
            return json.dumps(records, ensure_ascii=False, separators=(",", ":")) if records else None
        except Exception:
            failed = True
            logger.exception("[%s] Notebook recall failed; continuing without notes", session_id)
            return None
        finally:
            trace_event("scratchpad_context", "scratchpad_retrieval", session_id=session_id,
                        payload={"duration_ms": (time.perf_counter() - started) * 1000,
                                 "matched_ids": [entry["id"] for entry in entries], "failed": failed})
