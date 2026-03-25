import logging
from datetime import datetime

logger = logging.getLogger("context_builder")


class ContextBuilder:
    def __init__(
        self,
        system_prompt: str,
        user_context,
        history_store,
        # memory_store and memory_limit removed! Orchestrator handles this now.
        history_limit: int = 6,
        summary_store=None,
    ):
        self.system_prompt = system_prompt
        self.user_context = user_context or {}
        self.history_store = history_store
        self.history_limit = history_limit
        self.summary_store = summary_store

        logger.info(
            "ContextBuilder initialized (history_limit=%d, summary=%s)",
            history_limit,
            summary_store is not None,
        )

    def build(
        self,
        session_id: str,
        user_text: str,
        injected_context: str | None = None,
    ) -> list[dict]:
        logger.info("[%s] Building context", session_id)
        logger.debug("[%s] User input len=%d", session_id, len(user_text))

        messages: list[dict] = []

        # --------------------------------------------------
        # 1. Base system prompt
        # --------------------------------------------------
        messages.append({
            "role": "system",
            "content": self.system_prompt,
        })
        logger.debug("[%s] Added base system prompt", session_id)

        # --------------------------------------------------
        # 2. Current local system datetime
        # --------------------------------------------------
        now_local = datetime.now().astimezone()
        messages.append({
            "role": "system",
            "content": f"Current system datetime (local): {now_local.isoformat()}",
        })
        logger.debug("[%s] Added current system datetime context", session_id)

        # --------------------------------------------------
        # 3. Static user profile/context from config
        # --------------------------------------------------
        if self.user_context:
            user_lines = []
            for key, value in self.user_context.items():
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                if isinstance(value, str) and "\n" in value:
                    user_lines.append(f"- {key}:")
                    user_lines.extend(f"  {line}" for line in value.splitlines() if line.strip())
                else:
                    user_lines.append(f"- {key}: {value}")

            if user_lines:
                messages.append({
                    "role": "system",
                    "content": "User profile/context (configured):\n" + "\n".join(user_lines),
                })

        # --------------------------------------------------
        # 4. Injected Context (Memories & Tool Outputs)
        # --------------------------------------------------
        if injected_context:
            messages.append({
                "role": "system",
                "content": (
                    "BACKGROUND CONTEXT (Retrieved Memories & Tool Results):\n"
                    f"{injected_context}\n\n"
                    "Use the above background context to inform your response. "
                    "Blend this information naturally into the conversation. "
                    "Do not explicitly announce that you are reading from a database or memory log."
                ),
            })
            logger.info("[%s] Added injected context (len=%d)", session_id, len(injected_context))

        # --------------------------------------------------
        # 5. Conversation summary
        # --------------------------------------------------
        summary = self.summary_store.get(session_id) if self.summary_store else None
        if summary:
            messages.append({
                "role": "system",
                "content": f"Summary of previous conversation:\n{summary}",
            })

        # --------------------------------------------------
        # 6. Recent conversation history (deduplicated)
        # --------------------------------------------------
        history_limit = 2 if summary else self.history_limit

        history = self.history_store.get_recent(
            session_id=session_id,
            limit=history_limit,
        )

        added_history = 0
        seen = set()

        for row in history:
            role = row["role"]
            if role not in {"user", "assistant"}:
                continue

            content = row["content"].strip()
            if not content:
                continue

            key = (role, content)
            if key in seen:
                continue

            if role == "user" and content == user_text.strip():
                continue

            seen.add(key)
            added_history += 1

            messages.append({
                "role": role,
                "content": content,
            })

        # --------------------------------------------------
        # 7. Current user input (always last)
        # --------------------------------------------------
        messages.append({
            "role": "user",
            "content": user_text,
        })

        logger.debug("[%s] Final context built (total_messages=%d)", session_id, len(messages))
        print(messages)
        return messages