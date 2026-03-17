import logging

from app.config import Config
from app.llm.ollama_stream import OllamaClient
from app.core.orchestrator import Orchestrator
from app.storage.database import Database
from app.storage.vector_store import VectorStore  # NEW: Import the vector store
from app.memory.chat_history import ChatHistoryStore
from app.memory.memory_store import MemoryStore
from app.memory.summary_store import SummaryStore
from app.services.context_builder import ContextBuilder
from app.services.summarizer import HistorySummarizer
from app.tools.web_search import SearXNGClient
from app.services.search_summarizer import SearchResultSummarizer
from app.tools.web_search import WebSearchTool
from app.planners.factory import build_planner
from app.memory.memory_policy import SimpleMemoryPolicy
from app.services.tool_executor import ToolExecutor

logger = logging.getLogger("orchestrator_factory")


def build_orchestrator() -> Orchestrator:
    logger.info("Building orchestrator")

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------
    logger.info("Loading configuration")
    config = Config()

    logger.debug(
        "Config summary: llm_model=%s, tools=%s",
        config.llm.get("model"),
        list(config.tools.keys()),
    )

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------
    logger.info(
        "Initializing LLM client (model=%s, host=%s)",
        config.llm.get("model"),
        config.llm.get("host"),
    )

    llm = OllamaClient(
        model=config.llm["model"],
        host=config.llm["host"],
        options={
            "temperature": config.llm["generation"]["temperature"],
            "top_p": config.llm["generation"]["top_p"],
            "num_predict": config.llm["generation"]["max_tokens"],
        },
        timeout_s=config.llm.get("timeout_s", 30.0),
        max_retries=config.llm.get("max_retries", 2),
        retry_backoff_s=config.llm.get("retry_backoff_s", 0.25),
    )

    logger.debug(
        "LLM options: temperature=%.2f top_p=%.2f max_tokens=%d",
        config.llm["generation"]["temperature"],
        config.llm["generation"]["top_p"],
        config.llm["generation"]["max_tokens"],
    )

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------
    logger.info("Initializing database and stores")

    db = Database()
    vector_store = VectorStore()  # NEW: Initialize ChromaDB (CPU based)
    
    # UPDATED: Pass vector_store to the memory and history stores
    history_store = ChatHistoryStore(db, vector_store)
    memory_store = MemoryStore(db, vector_store)
    summary_store = SummaryStore(db)

    logger.debug("Storage initialized: database, vector_store, history, memory, summary")

    # --------------------------------------------------
    # Planner
    # --------------------------------------------------
    logger.info("Building planner")
    planner = build_planner(config, llm)
    logger.info("Planner ready: %s", planner.__class__.__name__)

    # --------------------------------------------------
    # Summarizers
    # --------------------------------------------------
    logger.info("Initializing summarizers")

    history_summarizer = HistorySummarizer(llm)
    search_summarizer = SearchResultSummarizer(llm)

    logger.debug(
        "Summarizers ready: history=%s search=%s",
        history_summarizer.__class__.__name__,
        search_summarizer.__class__.__name__,
    )

    # --------------------------------------------------
    # Memory policy
    # --------------------------------------------------
    memory_policy = SimpleMemoryPolicy()
    logger.debug("Memory policy: %s", memory_policy.__class__.__name__)

    # --------------------------------------------------
    # Tools
    # --------------------------------------------------
    tools = {}
    
    web_cfg = config.tools.get("web", {})

    if web_cfg.get("enabled", False):
        logger.info("Web search tool enabled via config")

        web_client = SearXNGClient(
            base_url=web_cfg.get("base_url", config.tools["web"]["base_url"]),
            timeout=web_cfg.get("timeout", config.planner["timeout_ms"] / 1000),
            max_retries=web_cfg.get("max_retries", 2),
            retry_backoff_s=web_cfg.get("retry_backoff_s", 0.25),
        )

        if web_client.probe():
            logger.info("Web search backend reachable")
        else:
            logger.warning("Web search backend unreachable")

        web_tool = WebSearchTool(
            client=web_client,
            summarizer=search_summarizer,
        )

        tools[web_tool.name] = web_tool
        logger.info("Web search tool registered as '%s'", web_tool.name)

    else:
        logger.info("Web search tool disabled via config")

    tool_executor = ToolExecutor(tools)

    # --------------------------------------------------
    # Context builder
    # --------------------------------------------------
    logger.info("Setting up context builder")

    # UPDATED: Removed memory_store and memory_limit since Orchestrator handles retrieval now
    context_builder = ContextBuilder(
        system_prompt=config.assistant["system_prompt"],
        user_context=config.user_context,
        history_store=history_store,
        summary_store=summary_store,
        history_limit=config.context["history_limit"],
    )

    logger.debug(
        "Context builder configured (history_limit=%d)",
        config.context["history_limit"],
    )

    # --------------------------------------------------
    # Orchestrator
    # --------------------------------------------------
    logger.info("Initializing orchestrator")

    orchestrator = Orchestrator(
        llm=llm,
        context_builder=context_builder,
        history_store=history_store,
        memory_store=memory_store,
        summary_store=summary_store,
        summarizer=history_summarizer,
        planner=planner,
        tool_executor=tool_executor,
        memory_policy=memory_policy,
        summary_trigger=config.orchestrator["summary_trigger"],
    )

    logger.info(
        "Orchestrator built successfully (tools=%d)",
        len(tools),
    )

    return orchestrator