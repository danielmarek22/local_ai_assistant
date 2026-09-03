import argparse
import json
import uuid

from app.config import Config
from app.logging import setup_logging_from_config
from app.memory.chat_history import ChatHistoryStore
from app.memory.memory_store import MemoryStore
from app.paths import DATA_DIR, STATIC_DIR
from app.storage.database import Database
from app.storage.vector_store import VectorStore
from app.ui.console import print_event


def reconcile_indexes(config: Config) -> dict[str, dict[str, int]]:
    """Rebuild both retrieval indexes from canonical SQLite records."""
    db = Database(
        path=str(DATA_DIR / "assistant.db"),
        legacy_local_human_id=config.local_human["id"],
        legacy_local_human_name=config.local_human["display_name"],
    )
    try:
        vector_store = VectorStore(path=str(DATA_DIR / "vectordb"))
        assistant_id = str(config.assistant.get("id", "default-agent")).strip()
        assistant_name = str(config.assistant.get("display_name", "Astra")).strip()
        history = ChatHistoryStore(
            db,
            vector_store,
            uploads_root=str(STATIC_DIR / "uploads"),
            local_human_id=config.local_human["id"],
            local_human_name=config.local_human["display_name"],
            local_assistant_id=assistant_id or "default-agent",
            local_assistant_name=assistant_name or "Astra",
        )
        memory = MemoryStore(db, vector_store)
        return {
            "semantic": memory.reconcile_index(),
            "episodic": history.reconcile_index(),
        }
    finally:
        db.conn.close()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Astra local assistant")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("chat", "reconcile-indexes"),
        default="chat",
        help="start the console chat or reconcile retrieval indexes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    config = Config()
    setup_logging_from_config(config.logging)

    if args.command == "reconcile-indexes":
        print(json.dumps(reconcile_indexes(config), sort_keys=True))
        return 0

    from app.core.orchestrator_factory import build_orchestrator

    orchestrator = build_orchestrator()
    session_id = uuid.uuid4().hex[:8]

    while True:
        user_text = input("\nYou: ")
        if user_text.strip().lower() in {"exit", "quit"}:
            break

        for event in orchestrator.handle_user_input(session_id, user_text):
            print_event(event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
