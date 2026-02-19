import logging

logger = logging.getLogger("context_builder")


class ContextBuilder:
    def __init__(
        self,
        system_prompt: str,
        user_context,
        history_store,
        memory_store,
        history_limit: int = 6,
        memory_limit: int = 5,
        summary_store=None,
    ):
        self.system_prompt = system_prompt
        self.user_context = user_context or {}
        self.history_store = history_store
        self.memory_store = memory_store
        self.history_limit = history_limit
        self.memory_limit = memory_limit
        self.summary_store = summary_store

        logger.info(
            "ContextBuilder initialized (history_limit=%d, memory_limit=%d, summary=%s)",
            history_limit,
            memory_limit,
            summary_store is not None,
        )

    def build(
        self,
        session_id: str,
        user_text: str,
        tool_context: str | None = None,
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
        # 2. Static user profile/context from config (optional)
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
                    "content": (
                        "User profile/context (configured):\n"
                        + "\n".join(user_lines)
                    ),
                })
                logger.info(
                    "[%s] Added configured user context (%d fields)",
                    session_id,
                    len(user_lines),
                )

        # --------------------------------------------------
        # 3. Tool-provided context (optional, system-level)
        # --------------------------------------------------
        if tool_context:
            messages.append({
                "role": "system",
                "content": tool_context,
            })
            logger.info(
                "[%s] Added tool context (len=%d)",
                session_id,
                len(tool_context),
            )
        else:
            logger.debug("[%s] No tool context provided", session_id)

        # --------------------------------------------------
        # 4. Relevant long-term memory
        # --------------------------------------------------
        memories = self.memory_store.get_relevant(
            query=user_text,
            limit=self.memory_limit,
        )

        if memories:
            logger.info(
                "[%s] Retrieved %d relevant memories",
                session_id,
                len(memories),
            )

            memory_block = (
                "The following information is known about the user "
                "and should be considered when responding:\n"
            )
            for m in memories:
                memory_block += f"- {m}\n"

            messages.append({
                "role": "system",
                "content": memory_block.strip(),
            })
        else:
            logger.debug("[%s] No relevant memories found", session_id)

        # --------------------------------------------------
        # 5. Conversation summary (if present)
        # --------------------------------------------------
        summary = self.summary_store.get(session_id) if self.summary_store else None
        if summary:
            messages.append({
                "role": "system",
                "content": (
                    "Summary of previous conversation:\n"
                    f"{summary}"
                ),
            })
            logger.info(
                "[%s] Added conversation summary (len=%d)",
                session_id,
                len(summary),
            )
        else:
            logger.debug("[%s] No conversation summary available", session_id)

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
                logger.debug("[%s] Skipping duplicate history entry", session_id)
                continue

            if role == "user" and content == user_text.strip():
                logger.debug("[%s] Skipping current input from history", session_id)
                continue

            seen.add(key)
            added_history += 1

            messages.append({
                "role": role,
                "content": content,
            })

        logger.info(
            "[%s] Added %d history messages (limit=%d)",
            session_id,
            added_history,
            history_limit,
        )

        # --------------------------------------------------
        # 7. Current user input (always last)
        # --------------------------------------------------
        messages.append({
            "role": "user",
            "content": user_text,
        })

        logger.debug(
            "[%s] Final context built (total_messages=%d)",
            session_id,
            len(messages),
        )

        return messages
