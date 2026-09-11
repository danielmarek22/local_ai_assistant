from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from app.core.session_ids import SessionDeletedError


class SessionTurnCoordinator:
    """Serializes a session and gives waiting user turns priority over events."""

    def __init__(self, global_concurrency: int = 1):
        if int(global_concurrency) < 1:
            raise ValueError("global_concurrency must be at least one")
        self._global_concurrency = int(global_concurrency)
        self._active_sessions: set[str] = set()
        self._condition = asyncio.Condition()
        self._waiting_users = 0
        self._closed_sessions: set[str] = set()

    def is_closed(self, session_id: str) -> bool:
        return session_id in self._closed_sessions

    async def close_session(self, session_id: str) -> None:
        """Reject new/queued turns, then drain active work without cancelling workers."""
        async with self._condition:
            self._closed_sessions.add(session_id)
            self._condition.notify_all()
            await self._condition.wait_for(lambda: session_id not in self._active_sessions)

    @asynccontextmanager
    async def user_turn(self, session_id: str):
        async with self._turn(session_id, is_user=True):
            yield

    @asynccontextmanager
    async def event_turn(self, session_id: str):
        async with self._turn(session_id, is_user=False):
            yield

    @asynccontextmanager
    async def _turn(self, session_id: str, *, is_user: bool):
        async with self._condition:
            if is_user:
                self._waiting_users += 1
            try:
                # Admit a turn only when both resources are available. Reserving
                # a session while waiting for capacity can block a priority user
                # behind an event that is itself waiting for that user to run.
                await self._condition.wait_for(
                    lambda: self.is_closed(session_id) or (
                        session_id not in self._active_sessions
                        and len(self._active_sessions) < self._global_concurrency
                        and (is_user or self._waiting_users == 0)
                    )
                )
                if self.is_closed(session_id):
                    raise SessionDeletedError()
            finally:
                if is_user:
                    self._waiting_users -= 1
                    self._condition.notify_all()
            self._active_sessions.add(session_id)

        try:
            yield
        finally:
            async with self._condition:
                self._active_sessions.remove(session_id)
                self._condition.notify_all()
