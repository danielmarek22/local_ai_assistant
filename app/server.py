from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json

from app.config import Config
from app.llm.ollama_stream import OllamaClient
from app.core.orchestrator import Orchestrator
from app.core.context_builder import ContextBuilder
from app.storage.database import Database
from app.memory.chat_history import ChatHistoryStore
from app.memory.memory_store import MemoryStore



app = FastAPI()


def create_orchestrator() -> Orchestrator:
    config = Config()

    db = Database()
    history_store = ChatHistoryStore(db)
    memory_store = MemoryStore(db)

    llm = OllamaClient(
        model=config.llm["model"],
        host=config.llm["host"],
        options={
            "temperature": config.llm["generation"]["temperature"],
            "top_p": config.llm["generation"]["top_p"],
            "num_predict": config.llm["generation"]["max_tokens"],
        },
    )

    context_builder = ContextBuilder(
        system_prompt=config.assistant["system_prompt"],
        history_store=history_store,
        memory_store=memory_store,
        history_limit=6,
        memory_limit=5,
    )

    return Orchestrator(
        llm=llm,
        context_builder=context_builder,
        history_store=history_store,
        memory_store=memory_store,
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    orchestrator = create_orchestrator()

    try:
        while True:
            user_text = await ws.receive_text()

            # Signal start
            await ws.send_text(json.dumps({
                "type": "assistant_start"
            }))

            # Stream assistant response
            for event in orchestrator.handle_user_input(user_text):
                if not event.is_final:
                    await ws.send_text(json.dumps({
                        "type": "assistant_chunk",
                        "content": event.text
                    }))
                    await asyncio.sleep(0)  # yield control

            # Signal end
            await ws.send_text(json.dumps({
                "type": "assistant_end"
            }))

    except WebSocketDisconnect:
        print("WebSocket disconnected")
