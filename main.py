from app.config import Config
from app.llm.ollama_stream import OllamaClient
from app.core.orchestrator import Orchestrator
from app.ui.console import print_event
from app.storage.database import Database
from app.memory.chat_history import ChatHistoryStore
from app.memory.memory_store import MemoryStore
from app.core.context_builder import ContextBuilder
from app.memory.summary_store import SummaryStore
from app.core.summarizer import HistorySummarizer
from app.core.planner import Planner
from app.tools.web_search import SearXNGClient
from app.tools.search_summarizer import SearchResultSummarizer



def main():
    config = Config()

    # --- Storage ---
    db = Database()
    history_store = ChatHistoryStore(db)
    memory_store = MemoryStore(db)
    summary_store = SummaryStore(db)
    planner = Planner()
    web_search = SearXNGClient(base_url="http://localhost:8080")
    
    # --- LLM ---
    llm = OllamaClient(
        model=config.llm["model"],
        host=config.llm["host"],
        options={
            "temperature": config.llm["generation"]["temperature"],
            "top_p": config.llm["generation"]["top_p"],
            "num_predict": config.llm["generation"]["max_tokens"],
        },
    )

    # --- Summarizer ---
    summarizer = HistorySummarizer(llm)
    search_summarizer = SearchResultSummarizer(llm)

    # --- Context builder ---
    context_builder = ContextBuilder(
        system_prompt=config.assistant["system_prompt"],
        history_store=history_store,
        memory_store=memory_store,
        summary_store=summary_store,
        history_limit=6,
        memory_limit=5,
    )

    # --- Orchestrator ---
    orchestrator = Orchestrator(
        llm=llm,
        context_builder=context_builder,
        history_store=history_store,
        memory_store=memory_store,
        summary_store=summary_store,
        summarizer=summarizer,
        planner=planner,
        web_search=web_search,
        search_summarizer=search_summarizer,
        summary_trigger=10,
    )

    # --- Main loop ---
    while True:
        user_text = input("\nYou: ")
        if user_text.strip().lower() in {"exit", "quit"}:
            break

        for event in orchestrator.handle_user_input(user_text):
            print_event(event)


if __name__ == "__main__":
    main()
