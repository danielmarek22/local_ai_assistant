from config import Config
from llm.ollama_stream import OllamaClient
from core.orchestrator import Orchestrator
from ui.console import print_event
from storage.database import Database
from memory.chat_history import ChatHistoryStore
from memory.memory_store import MemoryStore


def main():
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

    orchestrator = Orchestrator(
        llm=llm,
        system_prompt=config.assistant["system_prompt"],
        history_store=history_store,
        memory_store=memory_store,
    )

    while True:
        user_text = input("\nYou: ")
        if user_text.strip().lower() in {"exit", "quit"}:
            break

        for event in orchestrator.handle_user_input(user_text):
            print_event(event)


if __name__ == "__main__":
    main()
