import logging

from app.config import Config
from app.llm.ollama_stream import OllamaClient
from app.core.orchestrator import Orchestrator
from app.storage.database import Database
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
    history_store = ChatHistoryStore(db)
    memory_store = MemoryStore(db)
    summary_store = SummaryStore(db)

    logger.debug("Storage initialized: history, memory, summary")

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

    context_builder = ContextBuilder(
        system_prompt=config.assistant["system_prompt"],
        history_store=history_store,
        memory_store=memory_store,
        summary_store=summary_store,
        history_limit=config.context["history_limit"],
        memory_limit=config.context["memory_limit"],
    )

    logger.debug(
        "Context builder configured (history_limit=%d, memory_limit=%d)",
        config.context["history_limit"],
        config.context["memory_limit"],
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
