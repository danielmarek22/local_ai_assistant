from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from app.autonomy.broker import EventTurnOutcome, IntegrationEventBroker
from app.autonomy.coordinator import SessionTurnCoordinator
from app.autonomy.store import AutonomyStore, EventRecord
from app.core.events import (
    AssistantStateEvent,
    AutonomyOutcomeEvent,
)
from app.integrations import EventSpec, IntegrationEvent, IntegrationRegistry


logger = logging.getLogger("autonomy_runtime")
_SENTINEL = object()

OutputSink = Callable[[str, Any, str], Awaitable[None]]
NotificationSink = Callable[[str, dict[str, object], str], Awaitable[None]]
ApprovalProvider = Callable[[str, dict[str, object]], Awaitable[bool]]


class AutonomyRuntime:
    def __init__(
        self,
        orchestrator,
        registry: IntegrationRegistry,
        store: AutonomyStore,
        config: dict[str, object],
    ):
        self.orchestrator = orchestrator
        self.registry = registry
        self.store = store
        self.config = dict(config)
        self.coordinator = SessionTurnCoordinator(
            int(self.config.get("global_llm_concurrency", 1))
        )
        self.output_sink: OutputSink | None = None
        self.notification_sink: NotificationSink | None = None
        self.approval_provider: ApprovalProvider | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.broker = IntegrationEventBroker(
            registry=registry,
            store=store,
            handler=self._handle_event,
            enabled=bool(self.config.get("enabled", False)),
            max_queue_size=int(self.config.get("max_queue_size", 256)),
            max_chain_events=int(self.config.get("max_chain_events", 20)),
            max_chain_age_s=float(self.config.get("max_chain_age_s", 1800)),
            discard_handler=self._handle_discard,
        )

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self.broker.start()
        self.registry.start(self.broker.publish)

    async def close(self) -> None:
        try:
            self.registry.close()
        finally:
            try:
                await self.broker.close()
            finally:
                self.store.close()

    async def publish(self, event: IntegrationEvent) -> str:
        return await self.broker.publish_async(event)

    async def set_paused(self, paused: bool) -> None:
        await self.broker.set_paused(paused)

    def status(self) -> dict[str, object]:
        return self.broker.status()

    async def _handle_event(self, record: EventRecord, spec: EventSpec) -> EventTurnOutcome:
        event = record.event
        if event.session_id is None:
            raise RuntimeError("Cannot process an autonomous event without a target session")
        notification = None
        summary = "Event processed without an outcome."
        approval_callback = self._build_approval_callback(event.session_id)
        context = self.store.recent_context(
            event.session_id,
            int(self.config.get("recent_context_limit", 4000)),
        )
        async with self.coordinator.event_turn(event.session_id):
            iterator = self.orchestrator.handle_integration_event(
                session_id=event.session_id,
                event=event,
                spec=spec,
                autonomy_context=context,
                tool_approval_callback=approval_callback,
            )
            async for output in self._run_generator(iterator):
                if isinstance(output, AutonomyOutcomeEvent):
                    summary = output.summary
                    notification = output.notification
                    continue
                if isinstance(output, AssistantStateEvent) and self.output_sink is not None:
                    await self.output_sink(event.session_id, output, event.event_id)
        if notification is not None and self.notification_sink is not None:
            await self.notification_sink(event.session_id, notification, event.event_id)
        return EventTurnOutcome(summary=summary, notification=notification)

    async def _handle_discard(
        self,
        record: EventRecord,
        notification: dict[str, object],
    ) -> None:
        session_id = record.event.session_id
        if session_id is None:
            return
        message = str(notification.get("message", "")).strip()
        if message:
            self.orchestrator.history.add(session_id, "assistant", message)
        if self.notification_sink is not None:
            await self.notification_sink(session_id, notification, record.event.event_id)

    def _build_approval_callback(self, session_id: str):
        if self.approval_provider is None or self._loop is None:
            return None

        def request(request: dict[str, object]) -> bool:
            future = asyncio.run_coroutine_threadsafe(
                self.approval_provider(session_id, request),
                self._loop,
            )
            try:
                return bool(future.result(timeout=float(self.config.get("approval_timeout_s", 300)) + 5))
            except Exception:
                logger.exception("Autonomous approval failed for session %s", session_id)
                return False

        return request

    async def _run_generator(self, iterator: Iterator[Any]):
        loop = asyncio.get_running_loop()
        iterator = iter(iterator)
        while True:
            item = await loop.run_in_executor(None, self._next, iterator)
            if item is _SENTINEL:
                break
            yield item

    @staticmethod
    def _next(iterator):
        try:
            return next(iterator)
        except StopIteration:
            return _SENTINEL
