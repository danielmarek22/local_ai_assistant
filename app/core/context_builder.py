class ContextBuilder:
    def __init__(
        self,
        system_prompt: str,
        history_store,
        memory_store,
        history_limit: int = 6,
        memory_limit: int = 5,
    ):
        self.system_prompt = system_prompt
        self.history_store = history_store
        self.memory_store = memory_store
        self.history_limit = history_limit
        self.memory_limit = memory_limit

    def build(self, session_id: str, user_text: str) -> list[dict]:
        messages = []

        # 1. Base system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        # 2. Relevant memory
        memories = self.memory_store.get_relevant(
            query=user_text,
            limit=self.memory_limit
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
                "content": memory_block.strip()
            })

        # 3. Previous user messages only (no assistant, no duplicates)
        history = self.history_store.get_recent(
            session_id=session_id,
            limit=self.history_limit
        )

        seen = set()
        for row in history:
            if row["role"] != "user":
                continue

            content = row["content"].strip()

            # Skip duplicates
            if content in seen:
                continue

            # Skip current input if already stored
            if content == user_text.strip():
                continue

            seen.add(content)

            messages.append({
                "role": "user",
                "content": content
            })

        # 4. CURRENT user input (always last)
        messages.append({
            "role": "user",
            "content": user_text
        })

        print(messages)
        return messages
