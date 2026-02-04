from app.config import Config
from app.llm.ollama_stream import OllamaClient
from app.core.orchestrator import Orchestrator
from app.storage.database import Database
from app.memory.chat_history import ChatHistoryStore
from app.memory.memory_store import MemoryStore
from app.memory.summary_store import SummaryStore
from app.core.context_builder import ContextBuilder
from app.core.summarizer import HistorySummarizer
from app.tools.web_search import SearXNGClient
from app.tools.search_summarizer import SearchResultSummarizer
from app.core.planner_factory import build_planner
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_orchestrator() -> Orchestrator:
    logging.info("Loading configuration...")
    config = Config()

    # --- LLM ---
    logger.info("Initializing LLM client: %s", config.llm["model"])
    llm = OllamaClient(
        model=config.llm["model"],
        host=config.llm["host"],
        options={
            "temperature": config.llm["generation"]["temperature"],
            "top_p": config.llm["generation"]["top_p"],
            "num_predict": config.llm["generation"]["max_tokens"],
        },
    )

    # --- Storage ---
    logger.info("Initializing database and stores")
    db = Database()
    history_store = ChatHistoryStore(db)
    memory_store = MemoryStore(db)
    summary_store = SummaryStore(db)

    # --- Planner ---
    logger.info("Building planner")
    planner = build_planner(config, llm)

    # --- Tools ---
    web_search = SearXNGClient(base_url="http://localhost:8080")
    if web_search.probe():
        logger.info("Web search tool available")
    else:
        logger.warning("Web search tool unavailable, continuing without it")

    # --- Summarizers ---
    logger.info("Initializing summarizers")
    summarizer = HistorySummarizer(llm)
    search_summarizer = SearchResultSummarizer(llm)

    # --- Context builder ---
    logger.info("Setting up context builder")
    context_builder = ContextBuilder(
        system_prompt=config.assistant["system_prompt"],
        history_store=history_store,
        memory_store=memory_store,
        summary_store=summary_store,
        history_limit=6,
        memory_limit=5,
    )

    # --- Orchestrator ---
    logger.info("Orchestrator initialized successfully")
    return Orchestrator(
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
