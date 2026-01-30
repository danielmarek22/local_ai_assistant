from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from typing import Iterator, Any
import json
from app.core.orchestrator_factory import build_orchestrator
from app.logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

logger.info("Starting FastAPI server")


_SENTINEL = object()


def _next_or_sentinel(iterator: Iterator[Any]):
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


async def run_generator(gen: Iterator[Any]):
    loop = asyncio.get_running_loop()
    iterator = iter(gen)

    while True:
        item = await loop.run_in_executor(
            None, _next_or_sentinel, iterator
        )

        if item is _SENTINEL:
            break

        yield item


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    orchestrator = build_orchestrator()

    try:
        while True:
            user_text = await ws.receive_text()

            started = False

            async for event in run_generator(
                orchestrator.handle_user_input(user_text)
            ):
                if not started:
                    await ws.send_text(json.dumps({
                        "type": "assistant_start"
                    }))
                    started = True

                if not event.is_final:
                    await ws.send_text(json.dumps({
                        "type": "assistant_chunk",
                        "content": event.text
                    }))
                else:
                    await ws.send_text(json.dumps({
                        "type": "assistant_end",
                        "content": event.text
                    }))

    except WebSocketDisconnect:
        print("WebSocket disconnected")


@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

