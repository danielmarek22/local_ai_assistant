from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class SessionTurnCoordinator:
    """Serializes a session and gives waiting user turns priority over events."""

    def __init__(self, global_concurrency: int = 1):
        if int(global_concurrency) < 1:
            raise ValueError("global_concurrency must be at least one")
        self._global_concurrency = int(global_concurrency)
        self._active_global = 0
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._condition = asyncio.Condition()
        self._waiting_users = 0

    @asynccontextmanager
    async def user_turn(self, session_id: str):
        async with self._condition:
            self._waiting_users += 1
            self._condition.notify_all()
        waiting_registered = True
        try:
            async with self._session_lock(session_id):
                async with self._condition:
                    await self._condition.wait_for(
                        lambda: self._active_global < self._global_concurrency
                    )
                    self._waiting_users -= 1
                    waiting_registered = False
                    self._active_global += 1
                    self._condition.notify_all()
                try:
                    yield
                finally:
                    async with self._condition:
                        self._active_global -= 1
                        self._condition.notify_all()
        finally:
            if waiting_registered:
                async with self._condition:
                    self._waiting_users -= 1
                    self._condition.notify_all()

    @asynccontextmanager
    async def event_turn(self, session_id: str):
        async with self._condition:
            await self._condition.wait_for(lambda: self._waiting_users == 0)
        async with self._session_lock(session_id):
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._waiting_users == 0
                    and self._active_global < self._global_concurrency
                )
                self._active_global += 1
            try:
                yield
            finally:
                async with self._condition:
                    self._active_global -= 1
                    self._condition.notify_all()

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock
