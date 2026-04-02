import uuid

from app.config import Config
from app.core.orchestrator_factory import build_orchestrator
from app.logging import setup_logging_from_config
from app.ui.console import print_event


def main():
    config = Config()
    setup_logging_from_config(config.logging)
    orchestrator = build_orchestrator()
    session_id = uuid.uuid4().hex[:8]

    while True:
        user_text = input("\nYou: ")
        if user_text.strip().lower() in {"exit", "quit"}:
            break

        for event in orchestrator.handle_user_input(session_id, user_text):
            print_event(event)


if __name__ == "__main__":
    main()
