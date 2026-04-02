import logging
from datetime import datetime

from app.logging import trace_event
from app.perception.attachments import Attachment, ImageAttachment, attachment_from_payload, attachment_from_stored_record

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
        memory_context: str | None = None,
        tool_context: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> list[dict]:
        logger.info("[%s] Building context", session_id)
        attachments = attachments or []

        messages: list[dict] = []

        now_local = datetime.now().astimezone()

        summary_record = self.summary_store.get(session_id) if self.summary_store else None
        if isinstance(summary_record, tuple):
            summary = summary_record[0]
        else:
            summary = summary_record

        messages.append({
            "role": "system",
            "content": self._build_system_message(
                now_local_iso=now_local.isoformat(),
                memory_context=memory_context,
                tool_context=tool_context,
                summary=summary,
            ),
        })

        if memory_context or tool_context:
            logger.info(
                "[%s] Added injected context (memory=%s, tool=%s)", 
                session_id, 
                bool(memory_context), 
                bool(tool_context)
            )

        history_limit = 2 if summary else self.history_limit
        history = self.history_store.get_recent(
            session_id=session_id,
            limit=history_limit,
        )

        seen = set()
        current_user_key = self._build_seen_key("user", user_text.strip(), attachments)

        for row in history:
            role = row["role"]
            if role not in {"user", "assistant"}:
                continue

            content = row["content"].strip()
            if not content:
                continue

            row_attachments = self._normalize_history_attachments(row.get("attachments", []))
            key = self._build_seen_key(role, content, row_attachments)
            if key in seen:
                continue

            if role == "user" and key == current_user_key:
                continue

            seen.add(key)

            message = {
                "role": role,
                "content": content,
            }
            if role == "user" and row_attachments:
                message_images = [
                    attachment.to_llm_image()
                    for attachment in row_attachments
                    if isinstance(attachment, ImageAttachment)
                ]
                if message_images:
                    message["images"] = message_images

            messages.append(message)

        user_message = {
            "role": "user",
            "content": user_text,
        }
        if attachments:
            user_images = [
                attachment.to_llm_image()
                for attachment in attachments
                if isinstance(attachment, ImageAttachment)
            ]
            if user_images:
                user_message["images"] = user_images

        messages.append(user_message)

        logger.debug(
            "[%s] Final context built (total_messages=%d, current_images=%d)",
            session_id,
            len(messages),
            len(attachments),
        )
        trace_event(
            "context_builder",
            "context_built",
            session_id=session_id,
            payload={
                "user_text": user_text,
                "summary": summary,
                "memory_context": memory_context,
                "tool_context": tool_context,
                "history_limit_used": history_limit,
                "messages": messages,
            },
        )
        return messages

    def _build_system_message(
        self,
        now_local_iso: str,
        memory_context: str | None,
        tool_context: str | None,
        summary: str | None,
    ) -> str:
        sections = [
            self.system_prompt,
            f"Current system datetime (local): {now_local_iso}",
        ]

        user_context_section = self._build_user_context_section()
        if user_context_section:
            sections.append(user_context_section)

        # Assemble the background context internally
        combined_context_parts = []
        if memory_context:
            combined_context_parts.append(f"--- RETRIEVED MEMORY ---\n{memory_context}")
        if tool_context:
            combined_context_parts.append(f"--- TOOL RESULTS ---\n{tool_context}")

        if combined_context_parts:
            injected_context = "\n\n".join(combined_context_parts)
            sections.append(
                "BACKGROUND CONTEXT (Retrieved Memories & Tool Results):\n"
                f"{injected_context}\n\n"
                "Use the above background context to inform your response. "
                "Blend this information naturally into the conversation. "
                "Do not explicitly announce that you are reading from a database or memory log."
            )

        if summary:
            sections.append(f"Summary of previous conversation:\n{summary}")

        return "\n\n---\n\n".join(section for section in sections if section)

    def _build_user_context_section(self) -> str | None:
        if not self.user_context:
            return None

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

        if not user_lines:
            return None

        return "User profile/context (configured):\n" + "\n".join(user_lines)

    def _normalize_history_attachments(
        self,
        attachments: list[Attachment] | list[dict],
    ) -> list[Attachment]:
        normalized: list[Attachment] = []
        for attachment in attachments or []:
            if isinstance(attachment, Attachment):
                normalized.append(attachment)
                continue

            if isinstance(attachment, dict):
                if attachment.get("data") or attachment.get("base64_data"):
                    normalized.append(attachment_from_payload(attachment))
                    continue
                if attachment.get("storage_path") or attachment.get("url"):
                    normalized.append(attachment_from_stored_record(attachment))
        return normalized

    def _build_seen_key(
        self,
        role: str,
        content: str,
        attachments: list[Attachment],
    ) -> tuple:
        attachment_key = tuple(
            (
                attachment.attachment_id,
                attachment.sha256,
                attachment.storage_path,
                attachment.url,
                attachment.name,
            )
            for attachment in attachments
        )
        return role, content, attachment_key
