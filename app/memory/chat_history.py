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


logger = logging.getLogger("chat_history")


class ChatHistoryStore:
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        uploads_root: str = "static/uploads",
        image_summarizer=None,
    ):
        self.db = db
        self.vector_store = vector_store
        self.collection = self.vector_store.episodic_collection
        self.uploads_root = Path(uploads_root)
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self.image_summarizer = image_summarizer

    def add(
        self,
        session_id: str,
        role: str,
        content: str,
        attachments: list[Attachment] | None = None,
    ):
        current_time = time.time()
        attachments = attachments or []
        trace_event(
            "chat_history",
            "history_add",
            session_id=session_id,
            payload={
                "role": role,
                "content": content,
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
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        message_id = cursor.lastrowid

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

        vector_docs = [f"{role.upper()}: {content}"]
        vector_metadatas = [{
            "session_id": session_id,
            "role": role,
            "timestamp": current_time,
            "source": "message",
            "message_id": message_id,
        }]

        for record in attachment_records:
            if not record.summary_text:
                continue
            vector_docs.append(
                self._build_attachment_vector_doc(role, content, record)
            )
            vector_metadatas.append({
                "session_id": session_id,
                "role": role,
                "timestamp": current_time,
                "source": "image_attachment",
                "message_id": message_id,
                "attachment_id": record.attachment_id,
            })

        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in vector_docs],
            documents=vector_docs,
            metadatas=vector_metadatas,
        )

        return message_id

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
    ) -> str:
        parts = [
            f"{role.upper()} shared image '{attachment.name}'.",
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
            item["attachments"] = attachments_by_message.get(row["id"], [])
            hydrated_rows.append(item)

        return hydrated_rows

    def search_past_conversations(self, query: str, current_session: str, limit: int = 4, max_distance: float = 0.55) -> list[str]:
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
            SELECT id, role, content
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

    def get_all(self, session_id: str):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT id, role, content, timestamp
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
            SELECT
                ch.session_id,
                MIN(ch.timestamp) AS started_at,
                MAX(ch.timestamp) AS updated_at,
                COUNT(*) AS message_count,
                (
                    SELECT ch2.content
                    FROM chat_history ch2
                    WHERE ch2.session_id = ch.session_id
                    ORDER BY ch2.id ASC
                    LIMIT 1
                ) AS preview
            FROM chat_history ch
            GROUP BY ch.session_id
            ORDER BY updated_at DESC, ch.session_id DESC
            """
        )
        return cursor.fetchall()

    def delete_session(self, session_id: str) -> int:
        cursor = self.db.conn.cursor()
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

        return deleted_count
