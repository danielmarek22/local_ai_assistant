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

        # 2. Inject long-term memory (if any)
        memories = self.memory_store.get_all(limit=self.memory_limit)
        if memories:
            memory_block = "Relevant background information:\n"
            for m in memories:
                memory_block += f"- {m}\n"

            messages.append({
                "role": "system",
                "content": memory_block.strip()
            })

        # 3. Inject recent chat history (windowed)
        history = self.history_store.get_recent(
            session_id=session_id,
            limit=self.history_limit
        )

        for row in history:
            messages.append({
                "role": row["role"],
                "content": row["content"]
            })

        # 4. Current user input (always last)
        messages.append({
            "role": "user",
            "content": user_text
        })

        print(messages)
        return messages
