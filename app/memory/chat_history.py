import hashlib
import logging
import mimetypes
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from app.logging import trace_event
from app.perception.attachments import Attachment, ImageAttachment, attachment_from_stored_record
from app.storage.database import Database
from app.storage.session_lifecycle import is_session_deleted, require_writable_session
from app.storage.vector_store import VectorStore
from app.storage.vector_candidates import ranked_candidates
from app.core.conversation import (
    InputSource,
    SenderAttribution,
    SenderType,
    SessionKind,
)
from app.core.session_ids import validate_session_id
from app.paths import STATIC_DIR, resolve_app_path


logger = logging.getLogger("chat_history")


@dataclass(frozen=True)
class SessionDeletionResult:
    deleted_count: int
    cleanup_errors: tuple[str, ...] = ()

    @property
    def deleted(self) -> bool:
        return self.deleted_count > 0

    @property
    def cleanup_complete(self) -> bool:
        return not self.cleanup_errors


class ChatHistoryStore:
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        uploads_root: str = str(STATIC_DIR / "uploads"),
        image_summarizer=None,
        image_summary_timeout_s: float = 15.0,
        local_human_id: str = "local-human",
        local_human_name: str = "You",
        local_assistant_id: str = "default-agent",
        local_assistant_name: str = "Astra",
        episodic_max_distance: float = 0.70,
    ):
        self.db = db
        self.vector_store = vector_store
        self.collection = self.vector_store.episodic_collection
        self.uploads_root = resolve_app_path(uploads_root)
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self.image_summarizer = image_summarizer
        self.image_summary_timeout_s = max(0.1, float(image_summary_timeout_s))
        self.local_human_id = local_human_id
        self.local_human_name = local_human_name
        self.local_assistant_id = local_assistant_id
        self.local_assistant_name = local_assistant_name
        self.episodic_max_distance = float(episodic_max_distance)

    def ensure_session(self, session_id: str, kind: SessionKind | str = SessionKind.DIRECT) -> SessionKind:
        session_id = validate_session_id(session_id)
        requested_kind = SessionKind(kind)
        with self.db.transaction() as conn:
            require_writable_session(conn, session_id)
            row = conn.execute(
                "SELECT kind FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None:
                return SessionKind(row["kind"])
            conn.execute(
                "INSERT OR IGNORE INTO chat_sessions (session_id, kind) VALUES (?, ?)",
                (session_id, requested_kind.value),
            )
            row = conn.execute(
                "SELECT kind FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return SessionKind(row["kind"])

    def is_session_deleted(self, session_id: str) -> bool:
        with self.db.connection() as conn:
            return is_session_deleted(conn, validate_session_id(session_id))

    def get_session_kind(self, session_id: str) -> SessionKind:
        session_id = validate_session_id(session_id)
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT kind FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return SessionKind(row["kind"]) if row is not None else SessionKind.DIRECT

    def session_exists(self, session_id: str) -> bool:
        session_id = validate_session_id(session_id)
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT (
                    EXISTS(SELECT 1 FROM chat_sessions WHERE session_id = ?)
                    OR EXISTS(SELECT 1 FROM chat_history WHERE session_id = ?)
                ) AS session_exists
                """,
                (session_id, session_id),
            ).fetchone()
        return bool(row["session_exists"])

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

        with self.db.transaction() as conn:
            require_writable_session(conn, session_id)
            cursor = conn.cursor()
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
                ),
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
                    attachments,
                )

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

        vector_ids = [self._message_vector_id(message_id)] + [
            self._attachment_vector_id(record.attachment_id)
            for record in attachment_records
            if record.summary_text
        ]
        try:
            self.collection.upsert(
                ids=vector_ids,
                documents=vector_docs,
                metadatas=vector_metadatas,
            )
        except Exception:
            logger.exception(
                "Message %d was saved canonically but its episodic index update failed; "
                "reconciliation is required",
                message_id,
            )

        return message_id

    @staticmethod
    def _message_vector_id(message_id: int) -> str:
        return f"message:{message_id}"

    @staticmethod
    def _attachment_vector_id(attachment_id: int | None) -> str:
        if attachment_id is None:
            raise ValueError("Stored attachment is missing its canonical ID")
        return f"attachment:{attachment_id}"

    def reconcile_index(self, batch_size: int = 100) -> dict[str, int]:
        """Make the episodic index match canonical, visible chat history."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        records = self._canonical_vector_records()
        canonical_ids = {record["id"] for record in records}
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            self.collection.upsert(
                ids=[record["id"] for record in batch],
                documents=[record["document"] for record in batch],
                metadatas=[record["metadata"] for record in batch],
            )

        indexed_ids = set(self.collection.get(include=[])["ids"])
        orphaned_ids = sorted(indexed_ids - canonical_ids)
        if orphaned_ids:
            self.collection.delete(ids=orphaned_ids)

        report = {
            "canonical_count": len(records),
            "upserted_count": len(records),
            "removed_count": len(orphaned_ids),
        }
        logger.info("Reconciled episodic history index: %s", report)
        return report

    def _canonical_vector_records(self) -> list[dict]:
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT ch.id, ch.session_id, ch.role, ch.content, ch.timestamp,
                       ch.sender_id, ch.sender_display_name, ch.sender_type,
                       ch.input_source, COALESCE(cs.kind, 'direct') AS session_kind
                FROM chat_history ch
                LEFT JOIN chat_sessions cs ON cs.session_id = ch.session_id
                WHERE ch.excluded_from_context = 0
                ORDER BY ch.id ASC
                """
            ).fetchall()
            attachments = conn.execute(
                """
                SELECT ca.id, ca.message_id, ca.name, ca.mime_type, ca.storage_path,
                       ca.sha256, ca.size_bytes, ca.summary_text
                FROM chat_attachments ca
                JOIN chat_history ch ON ch.id = ca.message_id
                WHERE ch.excluded_from_context = 0
                  AND ca.summary_text IS NOT NULL AND ca.summary_text != ''
                ORDER BY ca.id ASC
                """
            ).fetchall()

        records = []
        messages = {}
        for row in rows:
            item = dict(row)
            sender = self.effective_sender(item)
            session_kind = SessionKind(item["session_kind"])
            messages[item["id"]] = (item, sender, session_kind)
            records.append({
                "id": self._message_vector_id(item["id"]),
                "document": self._build_message_vector_doc(
                    item["role"], item["content"], sender, session_kind
                ),
                "metadata": self._vector_metadata(item, sender, source="message"),
            })

        if not messages:
            return records

        for row in attachments:
            attachment_row = dict(row)
            item, sender, session_kind = messages[attachment_row["message_id"]]
            attachment = attachment_from_stored_record(attachment_row)
            records.append({
                "id": self._attachment_vector_id(attachment.attachment_id),
                "document": self._build_attachment_vector_doc(
                    item["role"], item["content"], attachment, sender, session_kind
                ),
                "metadata": self._vector_metadata(
                    item,
                    sender,
                    source="image_attachment",
                    attachment_id=attachment.attachment_id,
                ),
            })
        return records

    @staticmethod
    def _vector_metadata(
        message: dict,
        sender: SenderAttribution,
        *,
        source: str,
        attachment_id: int | None = None,
    ) -> dict:
        metadata = {
            "session_id": message["session_id"],
            "role": message["role"],
            "timestamp": str(message["timestamp"]),
            "source": source,
            "message_id": message["id"],
            "sender_id": sender.sender_id,
            "sender_display_name": sender.sender_display_name,
            "sender_type": sender.sender_type.value,
            "input_source": sender.input_source.value,
        }
        if attachment_id is not None:
            metadata["attachment_id"] = attachment_id
        return metadata

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
        attachments: list[Attachment],
    ) -> list[ImageAttachment]:
        message_dir = self._session_upload_dir(session_id) / str(message_id)
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
                summary_text=(
                    attachment.summary_text.strip()
                    if isinstance(attachment.summary_text, str)
                    and attachment.summary_text.strip()
                    else None
                ),
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

    def summarize_pending_attachments(self, message_id: int) -> int:
        if self.image_summarizer is None:
            return 0

        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT ca.id AS attachment_id, ca.name, ca.mime_type,
                       ca.storage_path, ca.sha256, ca.size_bytes, ca.summary_text,
                       ch.id, ch.session_id, ch.role, ch.content, ch.timestamp,
                       ch.sender_id, ch.sender_display_name, ch.sender_type,
                       ch.input_source, COALESCE(cs.kind, 'direct') AS session_kind
                FROM chat_attachments ca
                JOIN chat_history ch ON ch.id = ca.message_id
                LEFT JOIN chat_sessions cs ON cs.session_id = ch.session_id
                WHERE ca.message_id = ?
                  AND (ca.summary_text IS NULL OR ca.summary_text = '')
                ORDER BY ca.id ASC
                """,
                (message_id,),
            ).fetchall()

        deadline = time.monotonic() + self.image_summary_timeout_s
        completed = 0
        for row in rows:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                logger.warning(
                    "Image-summary budget exhausted for message %d; remaining images were skipped",
                    message_id,
                )
                break

            record = dict(row)
            attachment = attachment_from_stored_record({
                "id": record["attachment_id"],
                "name": record["name"],
                "mime_type": record["mime_type"],
                "storage_path": record["storage_path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "summary_text": record["summary_text"],
            })
            summary = self._summarize_attachment(
                attachment,
                record["content"],
                timeout_s=remaining_s,
            )
            if not summary:
                continue

            with self.db.transaction() as conn:
                update = conn.execute(
                    """
                    UPDATE chat_attachments SET summary_text = ?
                    WHERE id = ? AND (summary_text IS NULL OR summary_text = '')
                    """,
                    (summary, attachment.attachment_id),
                )
            if update.rowcount != 1:
                continue

            summarized = ImageAttachment(
                name=attachment.name,
                mime_type=attachment.mime_type,
                size_bytes=attachment.size_bytes,
                attachment_id=attachment.attachment_id,
                storage_path=attachment.storage_path,
                sha256=attachment.sha256,
                summary_text=summary,
            )
            sender = self.effective_sender(record)
            try:
                self.collection.upsert(
                    ids=[self._attachment_vector_id(summarized.attachment_id)],
                    documents=[self._build_attachment_vector_doc(
                        record["role"],
                        record["content"],
                        summarized,
                        sender,
                        SessionKind(record["session_kind"]),
                    )],
                    metadatas=[self._vector_metadata(
                        record,
                        sender,
                        source="image_attachment",
                        attachment_id=summarized.attachment_id,
                    )],
                )
            except Exception:
                logger.exception(
                    "Image attachment %d was summarized canonically but its episodic "
                    "index update failed; reconciliation is required",
                    summarized.attachment_id,
                )
            completed += 1

        return completed

    def _summarize_attachment(
        self,
        attachment: ImageAttachment,
        content: str,
        *,
        timeout_s: float,
    ) -> str | None:
        if self.image_summarizer is None:
            return None

        try:
            summary = self.image_summarizer.summarize(
                attachment,
                message_text=content,
                timeout_s=timeout_s,
            )
        except Exception:
            logger.exception("Failed to summarize image attachment '%s'", attachment.name)
            return None

        if not isinstance(summary, str):
            logger.warning("Image summarizer returned no text for '%s'", attachment.name)
            return None
        summary = summary.strip()
        if not summary:
            logger.warning("Image summarizer returned an empty result for '%s'", attachment.name)
            return None
        trace_event(
            "chat_history",
            "attachment_summary",
            payload={"attachment_name": attachment.name, "summary": summary},
        )
        return summary

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
            relative_path = (
                path.relative_to(STATIC_DIR)
                if path.is_absolute()
                else path.relative_to("static")
            )
        except ValueError:
            relative_path = path
        return f"/static/{relative_path.as_posix()}"

    def _load_attachments_for_message_ids(
        self,
        conn,
        message_ids: list[int],
    ) -> dict[int, list[Attachment]]:
        if not message_ids:
            return {}

        placeholders = ", ".join("?" for _ in message_ids)
        rows = conn.execute(
            f"""
            SELECT id, message_id, name, mime_type, storage_path, sha256, size_bytes, summary_text
            FROM chat_attachments
            WHERE message_id IN ({placeholders})
            ORDER BY id ASC
            """,
            message_ids,
        ).fetchall()

        attachments_by_message: dict[int, list[Attachment]] = {}
        for row in rows:
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

    def _rows_with_attachments(self, conn, rows):
        rows = list(rows)
        if not rows:
            return []

        attachments_by_message = self._load_attachments_for_message_ids(
            conn,
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

    def search_past_conversations(
        self,
        query: str,
        current_session: str,
        limit: int = 4,
        max_distance: float | None = None,
    ) -> list[str]:
        current_session = validate_session_id(current_session)
        if limit < 1:
            return []
        effective_max_distance = (
            self.episodic_max_distance
            if max_distance is None
            else float(max_distance)
        )
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where={"session_id": {"$ne": current_session}}
        )

        filtered_docs = []
        legacy_fallback = "I'm sorry, I lost my train of thought. Could you repeat that?"
        with self.db.connection() as conn:
            for vector_id, distance in ranked_candidates(results, limit):
                if distance > effective_max_distance:
                    continue
                doc = self._canonical_episodic_document(conn, vector_id, current_session)
                if doc is not None and legacy_fallback not in doc:
                    filtered_docs.append(doc)

        trace_event(
            "chat_history",
            "episodic_search",
            session_id=current_session,
            payload={
                "query": query,
                "limit": limit,
                "max_distance": effective_max_distance,
                "documents": filtered_docs,
            },
        )
        return filtered_docs

    def _canonical_episodic_document(self, conn, vector_id: str, current_session: str) -> str | None:
        source, separator, raw_id = vector_id.partition(":")
        if (not separator or source not in {"message", "attachment"}
                or not raw_id.isascii() or not raw_id.isdecimal() or len(raw_id) > 19):
            return None
        record_id = int(raw_id)
        if not 0 < record_id <= 2**63 - 1 or str(record_id) != raw_id:
            return None
        attachment = None
        message_id = record_id
        if source == "attachment":
            row = conn.execute(
                "SELECT * FROM chat_attachments WHERE id = ? AND summary_text IS NOT NULL AND summary_text != ''",
                (record_id,),
            ).fetchone()
            if row is None:
                return None
            attachment = attachment_from_stored_record(dict(row))
            message_id = row["message_id"]
        row = conn.execute(
            """SELECT ch.*, COALESCE(cs.kind, 'direct') AS session_kind
               FROM chat_history ch LEFT JOIN chat_sessions cs ON cs.session_id = ch.session_id
               WHERE ch.id = ? AND ch.session_id != ? AND ch.excluded_from_context = 0
                 AND NOT EXISTS (SELECT 1 FROM deleted_sessions ds WHERE ds.session_id = ch.session_id)""",
            (message_id, current_session),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        sender = self.effective_sender(item)
        kind = SessionKind(item["session_kind"])
        if attachment is not None:
            return self._build_attachment_vector_doc(item["role"], item["content"], attachment, sender, kind)
        return self._build_message_vector_doc(item["role"], item["content"], sender, kind)

    def get_recent(self, session_id: str, limit: int = 10):
        session_id = validate_session_id(session_id)
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, sender_id, sender_display_name, sender_type, input_source,
                       retry_error, retry_attempts
                FROM chat_history
                WHERE session_id = ? AND excluded_from_context = 0
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            hydrated = list(reversed(self._rows_with_attachments(conn, rows)))
        trace_event(
            "chat_history",
            "recent_history",
            session_id=session_id,
            payload={"limit": limit, "rows": hydrated},
        )
        return hydrated

    def get_summary_batch(self, session_id: str, after_message_id: int, limit: int = 100):
        """Read the next eligible messages in stable ID order, without a moving window."""
        session_id = validate_session_id(session_id)
        if after_message_id < 0 or limit < 1:
            raise ValueError("Summary checkpoint must be nonnegative and limit must be positive")
        with self.db.connection() as conn:
            rows = conn.execute(
                """SELECT id, role, content, sender_id, sender_display_name, sender_type, input_source
                   FROM chat_history
                   WHERE session_id = ? AND id > ? AND excluded_from_context = 0
                     AND role IN ('user', 'assistant')
                   ORDER BY id ASC LIMIT ?""",
                (session_id, after_message_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_messages(self, session_id: str) -> int:
        session_id = validate_session_id(session_id)
        with self.db.connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS message_count FROM chat_history
                   WHERE session_id = ? AND excluded_from_context = 0""",
                (session_id,),
            ).fetchone()
        return int(row["message_count"])

    def get_before(self, session_id: str, message_id: int, limit: int = 2):
        session_id = validate_session_id(session_id)
        if limit <= 0:
            return []
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, sender_id, sender_display_name, sender_type, input_source
                FROM chat_history
                WHERE session_id = ? AND id < ? AND excluded_from_context = 0
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, message_id, limit),
            ).fetchall()
        rows = list(reversed([dict(row) for row in rows]))
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
        session_id = validate_session_id(session_id)
        if limit <= 0:
            return []
        with self.db.connection() as conn:
            rows = conn.execute(
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
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all(self, session_id: str):
        session_id = validate_session_id(session_id)
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, timestamp, sender_id, sender_display_name, sender_type,
                       input_source, retry_error, retry_attempts
                FROM chat_history
                WHERE session_id = ? AND excluded_from_context = 0
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            return self._rows_with_attachments(conn, rows)

    def mark_turn_failed(self, session_id: str, user_message_id: int, message: str) -> int:
        session_id = validate_session_id(session_id)
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE chat_history
                SET retry_error = ?, retry_attempts = retry_attempts + 1
                WHERE id = ? AND session_id = ? AND role = 'user'
                """,
                (message, int(user_message_id), session_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Retry target is not a user message in this session")
            row = conn.execute(
                "SELECT retry_attempts FROM chat_history WHERE id = ?",
                (int(user_message_id),),
            ).fetchone()
        return int(row["retry_attempts"])

    def resolve_turn_failure(self, session_id: str, user_message_id: int) -> None:
        session_id = validate_session_id(session_id)
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE chat_history SET retry_error = NULL
                WHERE id = ? AND session_id = ? AND role = 'user'
                """,
                (int(user_message_id), session_id),
            )

    def get_retryable_user_message(self, session_id: str, user_message_id: int):
        session_id = validate_session_id(session_id)
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, timestamp, sender_id, sender_display_name, sender_type,
                       input_source, retry_error, retry_attempts
                FROM chat_history
                WHERE id = ? AND session_id = ? AND role = 'user'
                  AND retry_error IS NOT NULL AND excluded_from_context = 0
                """,
                (int(user_message_id), session_id),
            ).fetchall()
            rows = self._rows_with_attachments(conn, rows)
        return rows[0] if rows else None

    def list_sessions(self):
        with self.db.connection() as conn:
            return conn.execute(
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
                      AND ch2.excluded_from_context = 0
                    ORDER BY ch2.id ASC
                    LIMIT 1
                ) AS preview
            FROM session_ids sessions
            LEFT JOIN chat_sessions cs ON cs.session_id = sessions.session_id
            LEFT JOIN chat_history ch
              ON ch.session_id = sessions.session_id AND ch.excluded_from_context = 0
            GROUP BY sessions.session_id, cs.kind, cs.created_at, cs.updated_at
            HAVING COUNT(ch.id) > 0
            ORDER BY updated_at DESC, sessions.session_id DESC
                """
            ).fetchall()

    def delete_session(self, session_id: str) -> SessionDeletionResult:
        session_id = validate_session_id(session_id)
        with self.db.transaction() as conn:
            session_exists = bool(conn.execute(
                """
                SELECT (
                    EXISTS(SELECT 1 FROM chat_sessions WHERE session_id = ?)
                    OR EXISTS(SELECT 1 FROM chat_history WHERE session_id = ?)
                ) AS session_exists
                """,
                (session_id, session_id),
            ).fetchone()["session_exists"])
            if session_exists:
                conn.execute(
                    "INSERT OR IGNORE INTO deleted_sessions (session_id) VALUES (?)",
                    (session_id,),
                )
            # Canonical session-owned records disappear atomically with the fence.
            conn.execute("DELETE FROM conversation_summary WHERE session_id = ?", (session_id,))
            conn.execute(
                "DELETE FROM beliefs WHERE visibility = 'SESSION_CURRENT' AND scope_session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM belief_applications WHERE source_message_id IN "
                "(SELECT id FROM chat_history WHERE session_id = ?)", (session_id,),
            )
            conn.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM chat_attachments WHERE session_id = ?",
                (session_id,),
            )
            cursor = conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?",
                (session_id,),
            )
            deleted_count = cursor.rowcount

        cleanup_errors = []
        try:
            self.collection.delete(where={"session_id": session_id})
        except Exception:
            logger.exception("Failed to remove episodic index entries for session %s", session_id)
            cleanup_errors.append("vector_index")

        upload_dir = self._session_upload_dir(session_id)
        if upload_dir.exists():
            try:
                shutil.rmtree(upload_dir)
            except OSError:
                logger.exception("Failed to remove attachment files for session %s", session_id)
                cleanup_errors.append("attachments")

        canonical_count = deleted_count if deleted_count > 0 else int(session_exists)
        return SessionDeletionResult(canonical_count, tuple(cleanup_errors))

    def _session_upload_dir(self, session_id: str) -> Path:
        session_id = validate_session_id(session_id)
        upload_root = self.uploads_root.resolve()
        session_dir = (upload_root / session_id).resolve()
        if session_dir.parent != upload_root:
            raise ValueError("Invalid session ID: attachment path escapes upload root")
        return session_dir
