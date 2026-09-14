# Speech delivery

`AudioDelivery` owns the bounded synthesis queue and its worker. Constructing it does
not start work; the application lifespan calls `start()` and awaits `close()`.
Synthesis runs serially in the executor and returns completion or the backend error
to its caller. Full failure tracebacks are logged before delivery to the caller, so
exception cleanup cannot invalidate the live worker coroutine.

Closing stops new admission, drains work already admitted (including a producer
waiting for queue space), and waits for the worker. Repeated close calls share the
same drain task; cancellation of a shutdown caller does not abandon it. Cancellation
of a synthesis caller does not stop an already running blocking backend call.

This component has no FastAPI, WebSocket, or orchestrator dependency. Transport owns
text/audio presentation and fallback policy. Backend deadlines and a bounded wait
for stalled synthesis are still pending under Finding 37; this extraction preserves
the existing wait-for-synthesis behavior.
