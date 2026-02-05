class ContextBuilder:
    def __init__(
        self,
        system_prompt: str,
        history_store,
        memory_store,
        history_limit: int = 6,
        memory_limit: int = 5,
        summary_store=None,
    ):
        self.system_prompt = system_prompt
        self.history_store = history_store
        self.memory_store = memory_store
        self.history_limit = history_limit
        self.memory_limit = memory_limit
        self.summary_store = summary_store

    def build(
        self,
        session_id: str,
        user_text: str,
        tool_context: str | None = None,
    ) -> list[dict]:
        messages: list[dict] = []

        # --------------------------------------------------
        # 1. Base system prompt
        # --------------------------------------------------
        messages.append({
            "role": "system",
            "content": self.system_prompt,
        })

        # --------------------------------------------------
        # 2. Tool-provided context (optional, system-level)
        # --------------------------------------------------
        if tool_context:
            messages.append({
                "role": "system",
                "content": tool_context,
            })

        # --------------------------------------------------
        # 3. Relevant long-term memory
        # --------------------------------------------------
        memories = self.memory_store.get_relevant(
            query=user_text,
            limit=self.memory_limit,
        )

        if memories:
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

        # --------------------------------------------------
        # 4. Conversation summary (if present)
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

        # --------------------------------------------------
        # 5. Recent user history (deduplicated)
        # --------------------------------------------------
        history_limit = 2 if summary else self.history_limit

        history = self.history_store.get_recent(
            session_id=session_id,
            limit=history_limit,
        )

        seen = set()
        for row in history:
            if row["role"] != "user":
                continue

            content = row["content"].strip()

            if not content:
                continue

            # Skip duplicates
            if content in seen:
                continue

            # Skip current input if already stored
            if content == user_text.strip():
                continue

            seen.add(content)

            messages.append({
                "role": "user",
                "content": content,
            })

        # --------------------------------------------------
        # 6. Current user input (always last)
        # --------------------------------------------------
        messages.append({
            "role": "user",
            "content": user_text,
        })

        return messages
