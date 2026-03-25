import logging
from datetime import datetime

from app.perception.state import ImageAttachment

logger = logging.getLogger("context_builder")


class ContextBuilder:
    def __init__(
        self,
        system_prompt: str,
        user_context,
        history_store,
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
        attachments: list[ImageAttachment] | None = None,
    ) -> list[dict]:
        logger.info("[%s] Building context", session_id)
        logger.debug("[%s] User input len=%d", session_id, len(user_text))
        attachments = attachments or []

        messages: list[dict] = []

        messages.append({
            "role": "system",
            "content": self.system_prompt,
        })
        logger.debug("[%s] Added base system prompt", session_id)

        now_local = datetime.now().astimezone()
        messages.append({
            "role": "system",
            "content": f"Current system datetime (local): {now_local.isoformat()}",
        })
        logger.debug("[%s] Added current system datetime context", session_id)

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

        summary = self.summary_store.get(session_id) if self.summary_store else None
        if summary:
            messages.append({
                "role": "system",
                "content": f"Summary of previous conversation:\n{summary}",
            })

        history_limit = 2 if summary else self.history_limit

        history = self.history_store.get_recent(
            session_id=session_id,
            limit=history_limit,
        )

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

            messages.append({
                "role": role,
                "content": content,
            })

        user_message = {
            "role": "user",
            "content": user_text,
        }
        if attachments:
            user_message["images"] = [attachment.to_llm_image() for attachment in attachments]

        messages.append(user_message)

        logger.debug(
            "[%s] Final context built (total_messages=%d, current_images=%d)",
            session_id,
            len(messages),
            len(attachments),
        )
        return messages
