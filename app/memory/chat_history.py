import hashlib
import logging
import mimetypes
import shutil
import time
import uuid
from pathlib import Path

from app.logging import trace_event
from app.perception.attachments import Attachment, ImageAttachment, attachment_from_stored_record
from app.storage.database import Database
from app.storage.vector_store import VectorStore
from app.core.conversation import (
    InputSource,
    SenderAttribution,
    SenderType,
    SessionKind,
)


logger = logging.getLogger("chat_history")


class ChatHistoryStore:
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        uploads_root: str = "static/uploads",
        image_summarizer=None,
        local_human_id: str = "local-human",
        local_human_name: str = "You",
        local_assistant_id: str = "default-agent",
        local_assistant_name: str = "Astra",
    ):
        self.db = db
        self.vector_store = vector_store
        self.collection = self.vector_store.episodic_collection
        self.uploads_root = Path(uploads_root)
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self.image_summarizer = image_summarizer
        self.local_human_id = local_human_id
        self.local_human_name = local_human_name
        self.local_assistant_id = local_assistant_id
        self.local_assistant_name = local_assistant_name

    def ensure_session(self, session_id: str, kind: SessionKind | str = SessionKind.DIRECT) -> SessionKind:
        requested_kind = SessionKind(kind)
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT kind FROM chat_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row is not None:
            return SessionKind(row["kind"])
        cursor.execute(
            "INSERT OR IGNORE INTO chat_sessions (session_id, kind) VALUES (?, ?)",
            (session_id, requested_kind.value),
        )
        self.db.conn.commit()
        cursor.execute("SELECT kind FROM chat_sessions WHERE session_id = ?", (session_id,))
        return SessionKind(cursor.fetchone()["kind"])

    def get_session_kind(self, session_id: str) -> SessionKind:
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT kind FROM chat_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        return SessionKind(row["kind"]) if row is not None else SessionKind.DIRECT

    def session_exists(self, session_id: str) -> bool:
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT (
                EXISTS(SELECT 1 FROM chat_sessions WHERE session_id = ?)
                OR EXISTS(SELECT 1 FROM chat_history WHERE session_id = ?)
            ) AS session_exists
            """,
            (session_id, session_id),
        )
        return bool(cursor.fetchone()["session_exists"])

    def default_sender(self, role: str, input_source: InputSource | str | None = None) -> SenderAttribution:
        source = InputSource(input_source) if input_source is not None else None
        if role == "user":
            return SenderAttribution(
                self.local_human_id,
                self.local_human_name,
                SenderType.HUMAN,
                source or InputSource.LOCAL_TEXT,
            )
        if role == "assistant":
            return SenderAttribution(
                self.local_assistant_id,
                self.local_assistant_name,
                SenderType.LOCAL_ASSISTANT,
                source or InputSource.ASSISTANT_GENERATION,
            )
        if role == "tool":
            return SenderAttribution("tool-runtime", "Tool", SenderType.TOOL, source or InputSource.TOOL_RUNTIME)
        return SenderAttribution("system-runtime", "System", SenderType.SYSTEM, source or InputSource.SYSTEM_RUNTIME)

    def effective_sender(self, row: dict) -> SenderAttribution:
        fallback = self.default_sender(row["role"])
        try:
            return SenderAttribution(
                sender_id=row.get("sender_id") or fallback.sender_id,
                sender_display_name=row.get("sender_display_name") or fallback.sender_display_name,
                sender_type=SenderType(row.get("sender_type") or fallback.sender_type.value),
                input_source=InputSource(row.get("input_source") or fallback.input_source.value),
            )
        except ValueError:
            return fallback

    def add(
        self,
        session_id: str,
        role: str,
        content: str,
        attachments: list[Attachment] | None = None,
        sender: SenderAttribution | None = None,
        session_kind: SessionKind | str = SessionKind.DIRECT,
    ):
        current_time = time.time()
        attachments = attachments or []
        sender = sender or self.default_sender(role)
        session_kind = self.ensure_session(session_id, session_kind)
        trace_event(
            "chat_history",
            "history_add",
            session_id=session_id,
            payload={
                "role": role,
                "content": content,
                "sender_id": sender.sender_id,
                "sender_display_name": sender.sender_display_name,
                "sender_type": sender.sender_type.value,
                "input_source": sender.input_source.value,
                "attachments": [
                    {
                        "name": attachment.name,
                        "mime_type": attachment.mime_type,
                        "size_bytes": attachment.size_bytes,
                        "attachment_id": attachment.attachment_id,
                        "storage_path": attachment.storage_path,
                        "url": attachment.url,
                        "sha256": attachment.sha256,
                        "summary_text": attachment.summary_text,
                    }
                    for attachment in attachments
                ],
            },
        )

        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_history (
                session_id, role, content, sender_id, sender_display_name,
                sender_type, input_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, role, content, sender.sender_id, sender.sender_display_name,
                sender.sender_type.value, sender.input_source.value,
            )
        )
        message_id = cursor.lastrowid
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,),
        )

        attachment_records = []
        if attachments:
            attachment_records = self._store_attachments(
                cursor,
                session_id,
                message_id,
                role,
                content,
                attachments,
            )

        self.db.conn.commit()

        vector_docs = [self._build_message_vector_doc(role, content, sender, session_kind)]
        vector_metadatas = [{
            "session_id": session_id,
            "role": role,
            "timestamp": current_time,
            "source": "message",
            "message_id": message_id,
            "sender_id": sender.sender_id,
            "sender_display_name": sender.sender_display_name,
            "sender_type": sender.sender_type.value,
            "input_source": sender.input_source.value,
        }]

        for record in attachment_records:
            if not record.summary_text:
                continue
            vector_docs.append(
                self._build_attachment_vector_doc(role, content, record, sender, session_kind)
            )
            vector_metadatas.append({
                "session_id": session_id,
                "role": role,
                "timestamp": current_time,
                "source": "image_attachment",
                "message_id": message_id,
                "attachment_id": record.attachment_id,
                "sender_id": sender.sender_id,
                "sender_display_name": sender.sender_display_name,
                "sender_type": sender.sender_type.value,
                "input_source": sender.input_source.value,
            })

        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in vector_docs],
            documents=vector_docs,
            metadatas=vector_metadatas,
        )

        return message_id

    def _build_message_vector_doc(
        self,
        role: str,
        content: str,
        sender: SenderAttribution,
        session_kind: SessionKind,
    ) -> str:
        if session_kind == SessionKind.DIRECT:
            return f"{role.upper()}: {content}"
        return f"{sender.sender_type.value.upper()} {sender.sender_display_name}: {content}"

    def _store_attachments(
        self,
        cursor,
        session_id: str,
        message_id: int,
        role: str,
        content: str,
        attachments: list[Attachment],
    ) -> list[ImageAttachment]:
        message_dir = self.uploads_root / session_id / str(message_id)
        message_dir.mkdir(parents=True, exist_ok=True)
        stored_attachments: list[ImageAttachment] = []

        for attachment in attachments:
            if not isinstance(attachment, ImageAttachment):
                raise ValueError(
                    f"Unsupported attachment type for chat history storage: {attachment.__class__.__name__}"
                )

            payload = attachment.as_bytes()
            sha256 = hashlib.sha256(payload).hexdigest()
            file_path = message_dir / f"{sha256}{self._extension_for_mime_type(attachment.mime_type)}"
            if not file_path.exists():
                file_path.write_bytes(payload)
            trace_event(
                "chat_history",
                "attachment_stored",
                session_id=session_id,
                payload={
                    "file_path": str(file_path),
                    "bytes": len(payload),
                    "name": attachment.name,
                },
            )

            stored_attachment = ImageAttachment(
                name=attachment.name,
                mime_type=attachment.mime_type,
                size_bytes=attachment.size_bytes,
                storage_path=str(file_path),
                sha256=sha256,
                summary_text=self._summarize_attachment(attachment, content),
            )

            cursor.execute(
                """
                INSERT INTO chat_attachments (
                    message_id,
                    session_id,
                    name,
                    mime_type,
                    storage_path,
                    sha256,
                    size_bytes,
                    summary_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    stored_attachment.name,
                    stored_attachment.mime_type,
                    stored_attachment.storage_path,
                    stored_attachment.sha256,
                    stored_attachment.size_bytes,
                    stored_attachment.summary_text,
                ),
            )
            stored_attachments.append(
                ImageAttachment(
                    name=stored_attachment.name,
                    mime_type=stored_attachment.mime_type,
                    size_bytes=stored_attachment.size_bytes,
                    attachment_id=cursor.lastrowid,
                    storage_path=stored_attachment.storage_path,
                    url=self._public_url_for_storage_path(stored_attachment.storage_path),
                    sha256=stored_attachment.sha256,
                    summary_text=stored_attachment.summary_text,
                )
            )

        return stored_attachments

    def _summarize_attachment(self, attachment: ImageAttachment, content: str) -> str | None:
        if self.image_summarizer is None:
            return None

        try:
            summary = self.image_summarizer.summarize(attachment, message_text=content)
        except Exception:
            logger.exception("Failed to summarize image attachment '%s'", attachment.name)
            return None

        summary = summary.strip()
        trace_event(
            "chat_history",
            "attachment_summary",
            payload={"attachment_name": attachment.name, "summary": summary},
        )
        return summary or None

    def _build_attachment_vector_doc(
        self,
        role: str,
        content: str,
        attachment: ImageAttachment,
        sender: SenderAttribution | None = None,
        session_kind: SessionKind = SessionKind.DIRECT,
    ) -> str:
        sender = sender or self.default_sender(role)
        subject = role.upper()
        if session_kind == SessionKind.MANUAL_GROUP:
            subject = f"{sender.sender_type.value.upper()} {sender.sender_display_name}"
        parts = [
            f"{subject} shared image '{attachment.name}'.",
            f"Image summary: {attachment.summary_text}",
        ]
        if content and not (
            content.startswith("[User attached ") and content.endswith(" image]")
        ) and not (
            content.startswith("[User attached ") and content.endswith(" images]")
        ):
            parts.append(f"Related message text: {content}")
        return " ".join(parts)

    def _extension_for_mime_type(self, mime_type: str) -> str:
        extension = mimetypes.guess_extension(mime_type, strict=False) or ""
        if extension == ".jpe":
            return ".jpg"
        if extension:
            return extension
        return ".img"

    def _public_url_for_storage_path(self, storage_path: str) -> str:
        path = Path(storage_path)
        try:
            relative_path = path.relative_to(Path("static"))
        except ValueError:
            relative_path = path
        return f"/static/{relative_path.as_posix()}"

    def _load_attachments_for_message_ids(
        self,
        message_ids: list[int],
    ) -> dict[int, list[Attachment]]:
        if not message_ids:
            return {}

        placeholders = ", ".join("?" for _ in message_ids)
        cursor = self.db.conn.cursor()
        cursor.execute(
            f"""
            SELECT id, message_id, name, mime_type, storage_path, sha256, size_bytes, summary_text
            FROM chat_attachments
            WHERE message_id IN ({placeholders})
            ORDER BY id ASC
            """,
            message_ids,
        )

        attachments_by_message: dict[int, list[Attachment]] = {}
        for row in cursor.fetchall():
            attachment = attachment_from_stored_record({
                "id": row["id"],
                "name": row["name"],
                "mime_type": row["mime_type"],
                "storage_path": row["storage_path"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
                "url": self._public_url_for_storage_path(row["storage_path"]),
                "summary_text": row["summary_text"],
            })
            attachments_by_message.setdefault(row["message_id"], []).append(attachment)

        return attachments_by_message

    def _rows_with_attachments(self, rows):
        rows = list(rows)
        if not rows:
            return []

        attachments_by_message = self._load_attachments_for_message_ids(
            [row["id"] for row in rows]
        )

        hydrated_rows = []
        for row in rows:
            item = dict(row)
            sender = self.effective_sender(item)
            item.update({
                "sender_id": sender.sender_id,
                "sender_display_name": sender.sender_display_name,
                "sender_type": sender.sender_type.value,
                "input_source": sender.input_source.value,
            })
            item["attachments"] = attachments_by_message.get(row["id"], [])
            hydrated_rows.append(item)

        return hydrated_rows

    def search_past_conversations(self, query: str, current_session: str, limit: int = 4, max_distance: float = 0.65) -> list[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where={"session_id": {"$ne": current_session}}
        )

        if not results["documents"] or not results["documents"][0]:
            trace_event(
                "chat_history",
                "episodic_search",
                session_id=current_session,
                payload={"query": query, "limit": limit, "documents": []},
            )
            return []

        documents = results["documents"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else []
        
        filtered_docs = []

        # STRICT FILTERING: Drop episodic memories that are too far away
        for doc, distance in zip(documents, distances):
            if distance <= max_distance:
                filtered_docs.append(doc)
            else:
                logger.debug(f"Discarded episodic memory '{doc[:30]}...' (Distance: {distance:.3f} > {max_distance})")

        trace_event(
            "chat_history",
            "episodic_search",
            session_id=current_session,
            payload={"query": query, "limit": limit, "documents": filtered_docs},
        )
        return filtered_docs

    def get_recent(self, session_id: str, limit: int = 10):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT id, role, content, sender_id, sender_display_name, sender_type, input_source
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit)
        )
        rows = cursor.fetchall()
        hydrated = list(reversed(self._rows_with_attachments(rows)))
        trace_event(
            "chat_history",
            "recent_history",
            session_id=session_id,
            payload={"limit": limit, "rows": hydrated},
        )
        return hydrated

    def count_messages(self, session_id: str) -> int:
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) AS message_count FROM chat_history WHERE session_id = ?",
            (session_id,),
        )
        return int(cursor.fetchone()["message_count"])

    def get_before(self, session_id: str, message_id: int, limit: int = 2):
        if limit <= 0:
            return []
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT id, role, content, sender_id, sender_display_name, sender_type, input_source
            FROM chat_history
            WHERE session_id = ? AND id < ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, message_id, limit),
        )
        rows = list(reversed([dict(row) for row in cursor.fetchall()]))
        for row in rows:
            sender = self.effective_sender(row)
            row.update({
                "sender_id": sender.sender_id,
                "sender_display_name": sender.sender_display_name,
                "sender_type": sender.sender_type.value,
                "input_source": sender.input_source.value,
            })
        return rows

    def get_participant_senders_before(
        self,
        session_id: str,
        message_id: int,
        *,
        limit: int = 32,
    ) -> list[dict]:
        """Return the latest authoritative row for each prior participant sender."""
        if limit <= 0:
            return []
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            WITH latest AS (
                SELECT sender_id, MAX(id) AS latest_id
                FROM chat_history
                WHERE session_id = ? AND id < ?
                  AND sender_type IN ('human', 'external_agent')
                  AND sender_id IS NOT NULL AND sender_id != ''
                GROUP BY sender_id
            )
            SELECT ch.sender_id, ch.sender_display_name, ch.sender_type, latest.latest_id
            FROM latest
            JOIN chat_history ch ON ch.id = latest.latest_id
            ORDER BY latest.latest_id DESC, ch.sender_id ASC
            LIMIT ?
            """,
            (session_id, message_id, int(limit)),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all(self, session_id: str):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT id, role, content, timestamp, sender_id, sender_display_name, sender_type, input_source
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )
        return self._rows_with_attachments(cursor.fetchall())

    def list_sessions(self):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            WITH session_ids AS (
                SELECT session_id FROM chat_sessions
                UNION
                SELECT DISTINCT session_id FROM chat_history
            )
            SELECT
                sessions.session_id,
                COALESCE(cs.kind, 'direct') AS kind,
                COALESCE(MIN(ch.timestamp), cs.created_at) AS started_at,
                COALESCE(MAX(ch.timestamp), cs.updated_at) AS updated_at,
                COUNT(ch.id) AS message_count,
                (
                    SELECT ch2.content
                    FROM chat_history ch2
                    WHERE ch2.session_id = sessions.session_id
                    ORDER BY ch2.id ASC
                    LIMIT 1
                ) AS preview
            FROM session_ids sessions
            LEFT JOIN chat_sessions cs ON cs.session_id = sessions.session_id
            LEFT JOIN chat_history ch ON ch.session_id = sessions.session_id
            GROUP BY sessions.session_id, cs.kind, cs.created_at, cs.updated_at
            HAVING COUNT(ch.id) > 0
            ORDER BY updated_at DESC, sessions.session_id DESC
            """
        )
        return cursor.fetchall()

    def delete_session(self, session_id: str) -> int:
        cursor = self.db.conn.cursor()
        session_exists = self.session_exists(session_id)
        cursor.execute(
            "DELETE FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        cursor.execute(
            """
            DELETE FROM chat_attachments
            WHERE session_id = ?
            """,
            (session_id,)
        )
        cursor.execute(
            """
            DELETE FROM chat_history
            WHERE session_id = ?
            """,
            (session_id,)
        )
        deleted_count = cursor.rowcount
        self.db.conn.commit()

        self.collection.delete(where={"session_id": session_id})
        shutil.rmtree(self.uploads_root / session_id, ignore_errors=True)

        return deleted_count if deleted_count > 0 else int(session_exists)
