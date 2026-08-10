from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from app.autonomy.store import AutonomyStore, EventRecord
from app.integrations import EventSpec, IntegrationEvent, IntegrationRegistry


logger = logging.getLogger("autonomy_broker")


@dataclass(frozen=True)
class EventTurnOutcome:
    summary: str
    notification: dict[str, object] | None = None


EventHandler = Callable[[EventRecord, EventSpec], Awaitable[EventTurnOutcome]]
DiscardHandler = Callable[[EventRecord, dict[str, object]], Awaitable[None]]


class IntegrationEventBroker:
    def __init__(
        self,
        registry: IntegrationRegistry,
        store: AutonomyStore,
        handler: EventHandler,
        *,
        enabled: bool = True,
        max_queue_size: int = 256,
        max_chain_events: int = 20,
        max_chain_age_s: float = 1800.0,
        discard_handler: DiscardHandler | None = None,
    ):
        self.registry = registry
        self.store = store
        self.handler = handler
        self.enabled = bool(enabled)
        self.max_chain_events = max(1, int(max_chain_events))
        self.max_chain_age_s = max(1.0, float(max_chain_age_s))
        self.discard_handler = discard_handler
        self._queue: asyncio.PriorityQueue[tuple[int, str, str]] = asyncio.PriorityQueue(
            maxsize=max(1, int(max_queue_size))
        )
        self._queued: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: asyncio.Task | None = None
        self._reconciler: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._worker is not None:
            return
        self._loop = asyncio.get_running_loop()
        replayed, failed = self.store.recover_interrupted()
        if replayed or failed:
            logger.warning("Recovered autonomy journal (replayed=%d, failed=%d)", replayed, failed)
        self._stopping = False
        self._worker = asyncio.create_task(self._run(), name="autonomy-event-worker")
        self._reconciler = asyncio.create_task(self._reconcile(), name="autonomy-event-reconciler")
        await self._enqueue_pending()

    async def close(self) -> None:
        self._stopping = True
        tasks = [task for task in (self._worker, self._reconciler) if task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._worker = None
        self._reconciler = None

    def publish(self, event: IntegrationEvent) -> str:
        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("Autonomy event broker is not running")
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            raise RuntimeError("Use publish_async from the broker event loop")
        future = asyncio.run_coroutine_threadsafe(self.publish_async(event), loop)
        return future.result(timeout=10.0)

    async def publish_async(self, event: IntegrationEvent) -> str:
        spec = self.registry.validate_event(event)
        if event.session_id is None and event.correlation_id:
            operation = self.store.get_operation(event.correlation_id)
            if operation is not None:
                event = IntegrationEvent(
                    event=event.event, payload=event.payload, session_id=operation.session_id,
                    event_id=event.event_id, occurred_at=event.occurred_at,
                    correlation_id=event.correlation_id,
                    causation_id=event.causation_id or operation.event_id,
                    root_event_id=event.root_event_id or operation.root_event_id,
                    deduplication_key=event.deduplication_key, attachments=event.attachments,
                )
                spec = self.registry.validate_event(event)
        event_id = self.store.append_event(event, spec)
        if self.enabled and event.session_id is not None and not self.store.is_paused():
            await self._enqueue(event_id, spec.priority)
        return event_id

    async def set_paused(self, paused: bool) -> None:
        self.store.set_paused(paused)
        if not paused:
            await self._enqueue_pending()

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "paused": self.store.is_paused(),
            "queued": self._queue.qsize(),
            "running": self._worker is not None and not self._worker.done(),
        }

    async def _run(self) -> None:
        while True:
            priority, occurred_at, event_id = await self._queue.get()
            self._queued.discard(event_id)
            try:
                if self.store.is_paused() or not self.enabled:
                    continue
                record = self.store.claim_event(event_id)
                if record is None:
                    continue
                spec = self.registry.get_event_spec(record.event.event)
                if spec is None:
                    self.store.fail_event(event_id, "Event specification is no longer registered")
                    continue
                if self._chain_exceeded(record):
                    notification = {
                        "message": (
                            "I stopped an autonomous activity chain because it reached its "
                            "configured count or time limit."
                        ),
                        "delivery": "text",
                    }
                    self.store.fail_event(
                        event_id,
                        "Autonomous causal chain exceeded its configured count or time budget",
                        status="discarded",
                        notification=notification,
                    )
                    if self.discard_handler is not None:
                        await self.discard_handler(record, notification)
                    continue
                try:
                    outcome = await self.handler(record, spec)
                except Exception as exc:
                    logger.exception("Autonomous event %s failed", event_id)
                    self.store.fail_event(event_id, str(exc))
                else:
                    self.store.complete_event(event_id, outcome.summary, outcome.notification)
            finally:
                self._queue.task_done()

    async def _reconcile(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self.enabled and not self.store.is_paused():
                await self._enqueue_pending()

    async def _enqueue_pending(self) -> None:
        for event_id in self.store.pending_event_ids():
            record = self.store.get_event(event_id)
            if record is not None:
                await self._enqueue(event_id, record.priority)

    async def _enqueue(self, event_id: str, priority: int) -> None:
        if event_id in self._queued:
            return
        record = self.store.get_event(event_id)
        if record is None or record.status != "pending":
            return
        item = (priority, record.event.occurred_at.isoformat(), event_id)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("Autonomy wake queue full; event %s remains durable", event_id)
            return
        self._queued.add(event_id)

    def _chain_exceeded(self, record: EventRecord) -> bool:
        root_id = record.event.root_event_id or record.event.event_id
        count, started_at = self.store.chain_stats(root_id)
        if count > self.max_chain_events:
            return True
        if started_at is None:
            return False
        age = (datetime.now(timezone.utc) - started_at.astimezone(timezone.utc)).total_seconds()
        return age > self.max_chain_age_s
