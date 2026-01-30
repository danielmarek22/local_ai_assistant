from app.core.orchestrator_factory import build_orchestrator
from app.ui.console import print_event


def main():
    orchestrator = build_orchestrator()

    while True:
        user_text = input("\nYou: ")
        if user_text.strip().lower() in {"exit", "quit"}:
            break

        for event in orchestrator.handle_user_input(user_text):
            print_event(event)


if __name__ == "__main__":
    main()
