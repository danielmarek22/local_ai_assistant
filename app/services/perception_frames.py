from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re
import time
from typing import Callable
import uuid

from app.integrations import EventAttachmentRef, EventId, IntegrationEvent
from app.paths import STATIC_DIR
from app.perception.attachments import Attachment, ImageAttachment, attachment_from_payload
from app.perception.keys import PerceptionKey
from app.services.websocket_protocol import VisionFrame


logger = logging.getLogger("server")

VOICE_ATTACHMENT_FRAME_TYPE = "user_attached_frame"
_DEFAULT_DETECTION_INTERVAL_SECONDS = 5.0


class PerceptionFrameController:
    """Handle one connection's screen, webcam, and voice-adjacent image frames."""

    def __init__(
        self,
        *,
        orchestrator,
        watchdog,
        session_id: str,
        connection_id: str,
        event_attachment_root: Path = STATIC_DIR / "uploads" / "events",
        detection_interval_seconds: float = _DEFAULT_DETECTION_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        event_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ):
        self.orchestrator = orchestrator
        self.watchdog = watchdog
        self.session_id = session_id
        self.connection_id = connection_id
        self.event_attachment_root = event_attachment_root
        self.detection_interval_seconds = detection_interval_seconds
        self.clock = clock
        self.event_id_factory = event_id_factory
        self._last_detection = {"screen": 0.0, "webcam": 0.0}
        self._last_hash: dict[str, str | None] = {"screen": None, "webcam": None}

    async def handle(self, frame: VisionFrame) -> Attachment | None:
        key, source = self._frame_route(frame.type)
        if source is None:
            return None

        attachment = attachment_from_payload(frame.attachment)
        base64_data = getattr(attachment, "base64_data", None)
        image_hash = attachment.sha256 or self._fallback_hash(frame.attachment)

        perception_payload = {
            **attachment.to_perception_payload(),
            "sha256": image_hash,
            "source": source,
        }
        if base64_data:
            perception_payload["base64_data"] = base64_data

        if key is not None:
            self.orchestrator.perception.update(key, perception_payload)

        if frame.type == VOICE_ATTACHMENT_FRAME_TYPE:
            return attachment

        if image_hash == self._last_hash[source]:
            return None
        self._last_hash[source] = image_hash

        now = self.clock()
        if now - self._last_detection[source] < self.detection_interval_seconds:
            return None

        evaluate = self._evaluator(source)
        if evaluate is None:
            logger.debug(
                "[%s] Vision watchdog unavailable; stored %s perception only",
                self.connection_id,
                source,
            )
            return None
        if not base64_data:
            return None

        self._last_detection[source] = now
        if not await evaluate(base64_data):
            return None

        runtime = getattr(self.orchestrator, "autonomy_runtime", None)
        if runtime is None:
            logger.warning(
                "[%s] Vision event ignored because autonomy is unavailable",
                self.connection_id,
            )
            return None

        event_id = self.event_id_factory()
        event_attachment = self._persist_event_attachment(event_id, attachment)
        await runtime.publish(IntegrationEvent(
            event=EventId("vision", "attention_detected"),
            event_id=event_id,
            session_id=self.session_id,
            payload={
                "source": source,
                "description": self._event_description(source),
                "sha256": image_hash,
            },
            deduplication_key=f"{source}:{image_hash}",
            attachments=(event_attachment,),
        ))
        logger.info(
            "[%s] Vision watchdog published autonomous %s event",
            self.connection_id,
            source,
        )
        return None

    @staticmethod
    def _frame_route(frame_type: str) -> tuple[PerceptionKey | None, str | None]:
        if frame_type == "screen_frame":
            return PerceptionKey.SCREEN_SCENE, "screen"
        if frame_type == "webcam_frame":
            return PerceptionKey.WEBCAM_SCENE, "webcam"
        if frame_type == VOICE_ATTACHMENT_FRAME_TYPE:
            return None, "voice_attachment"
        return None, None

    def _evaluator(self, source: str):
        if self.watchdog is None:
            return None
        return (
            self.watchdog.evaluate_screen
            if source == "screen"
            else self.watchdog.evaluate_webcam
        )

    @staticmethod
    def _fallback_hash(payload: dict[str, object]) -> str:
        raw_base64 = payload.get("base64_data") or payload.get("data") or ""
        return hashlib.sha256(str(raw_base64).encode("utf-8")).hexdigest()

    @staticmethod
    def _event_description(source: str) -> str:
        if source == "screen":
            return (
                "The local screen watchdog detected a clear visual event in the "
                "latest screenshot. Proactively help the user, briefly and concretely."
            )
        return (
            "The local webcam watchdog detected that the user may need attention. "
            "Proactively check in briefly and helpfully."
        )

    def _persist_event_attachment(
        self,
        event_id: str,
        attachment: ImageAttachment,
    ) -> EventAttachmentRef:
        event_dir = self.event_attachment_root / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        suffix = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }.get(attachment.mime_type, ".bin")
        safe_stem = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(attachment.name).stem)
            or "attachment"
        )
        path = event_dir / f"{safe_stem}{suffix}"
        payload = attachment.as_bytes()
        path.write_bytes(payload)
        return EventAttachmentRef(
            name=attachment.name,
            mime_type=attachment.mime_type,
            storage_path=str(path),
            sha256=attachment.sha256 or hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
