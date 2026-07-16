import logging

from app.config import Config
from app.llm.ollama_stream import OllamaClient
from app.core.orchestrator import Orchestrator
from app.storage.database import Database
from app.storage.vector_store import VectorStore
from app.memory.chat_history import ChatHistoryStore
from app.memory.memory_store import MemoryStore
from app.memory.summary_store import SummaryStore
from app.services.context_builder import ContextBuilder
from app.services.image_summarizer import ImageSummarizer
from app.services.summarizer import HistorySummarizer
from app.tools.web_search import SearXNGClient
from app.services.search_summarizer import SearchResultSummarizer
from app.tools.web_search import WebSearchTool
from app.tools.bash_execution import BashExecutionTool
from app.tools.memory_write import MemoryWriteTool
from app.memory.memory_policy import SimpleMemoryPolicy
from app.services.memory_action_handler import MemoryActionHandler
from app.services.memory_retriever import MemoryRetriever
from app.services.tool_executor import ToolExecutor
from app.services.turn_finalizer import TurnFinalizer
from app.services.avatar_controls import (
    build_prompt_with_avatar_controls,
    discover_gesture_catalog,
)
from app.services.tool_controls import build_prompt_with_tools

logger = logging.getLogger("orchestrator_factory")


def _build_ollama_options(raw_generation: dict | None) -> dict:
    generation = raw_generation if isinstance(raw_generation, dict) else {}
    options: dict[str, object] = {}

    field_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "max_tokens": "num_predict",
        "num_predict": "num_predict",
        "rep_pen": "repeat_penalty",
        "repeat_penalty": "repeat_penalty",
        "num_ctx": "num_ctx",
    }

    for source_key, target_key in field_map.items():
        if source_key not in generation:
            continue
        options[target_key] = generation[source_key]

    return options


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

    llm_generation = config.llm.get("generation", {})
    llm_options = _build_ollama_options(llm_generation)
    thinking_cfg = config.llm.get("thinking", {})
    thinking_generation = thinking_cfg.get("generation", {}) if isinstance(thinking_cfg, dict) else {}
    thinking_options = _build_ollama_options(thinking_generation)

    llm = OllamaClient(
        model=config.llm["model"],
        host=config.llm["host"],
        options=llm_options,
        thinking_enabled=bool(thinking_cfg.get("enabled", False)) if isinstance(thinking_cfg, dict) else False,
        thinking_level=thinking_cfg.get("level") if isinstance(thinking_cfg, dict) else None,
        thinking_options=thinking_options,
        timeout_s=config.llm.get("timeout_s", 30.0),
        max_retries=config.llm.get("max_retries", 2),
        retry_backoff_s=config.llm.get("retry_backoff_s", 0.25),
    )

    logger.debug(
        "LLM options: temperature=%.2f top_p=%.2f top_k=%s max_tokens=%d rep_pen=%s",
        llm_generation.get("temperature", 0.0),
        llm_generation.get("top_p", 0.0),
        llm_generation.get("top_k"),
        llm_generation.get("max_tokens", llm_generation.get("num_predict", 0)),
        llm_generation.get("rep_pen", llm_generation.get("repeat_penalty")),
    )

    llm.preload()

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------
    logger.info("Initializing database and stores")

    db = Database()
    vector_store = VectorStore()

    history_store = ChatHistoryStore(db, vector_store)
    memory_store = MemoryStore(db, vector_store)
    summary_store = SummaryStore(db)

    logger.debug("Storage initialized: database, vector_store, history, memory, summary")

    # --------------------------------------------------
    # Summarizers
    # --------------------------------------------------
    logger.info("Initializing summarizers")

    history_summarizer = HistorySummarizer(llm)
    image_summarizer = ImageSummarizer(llm)
    search_summarizer = SearchResultSummarizer(llm)
    history_store.image_summarizer = image_summarizer

    logger.debug(
        "Summarizers ready: history=%s image=%s search=%s",
        history_summarizer.__class__.__name__,
        image_summarizer.__class__.__name__,
        search_summarizer.__class__.__name__,
    )

    # --------------------------------------------------
    # Memory policy
    # --------------------------------------------------
    memory_policy = SimpleMemoryPolicy()
    logger.debug("Memory policy: %s", memory_policy.__class__.__name__)

    memory_retriever = MemoryRetriever(
        memory_store=memory_store,
        history_store=history_store,
    )
    memory_action_handler = MemoryActionHandler(
        memory_store=memory_store,
        memory_policy=memory_policy,
    )
    turn_finalizer = TurnFinalizer(
        history_store=history_store,
        summary_store=summary_store,
        summarizer=history_summarizer,
        summary_trigger=config.orchestrator["summary_trigger"],
    )

    # --------------------------------------------------
    # Tools
    # --------------------------------------------------
    tools = {}
    
    web_cfg = config.tools.get("web", {})

    if web_cfg.get("enabled", False):
        logger.info("Web search tool enabled via config")

        web_client = SearXNGClient(
            base_url=web_cfg.get("base_url", config.tools["web"]["base_url"]),
            timeout=web_cfg.get("timeout", 10.0),  # Defaulted to 10s if missing
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

        # Inject Bash Execution Tool
    bash_tool = BashExecutionTool(timeout=15)
    tools[bash_tool.name] = bash_tool
    logger.info("Bash execution tool registered as '%s'", bash_tool.name)

    memory_write_tool = MemoryWriteTool(memory_action_handler=memory_action_handler)
    tools[memory_write_tool.name] = memory_write_tool
    logger.info("Memory write tool registered as '%s'", memory_write_tool.name)

    tool_executor = ToolExecutor(tools)

    # --------------------------------------------------
    # Context builder
    # --------------------------------------------------
    logger.info("Setting up context builder")
    gesture_catalog = discover_gesture_catalog()
    avatar_controls_cfg = config.assistant.get("avatar_controls", {})
    allowed_expressions = avatar_controls_cfg.get("expressions") if isinstance(avatar_controls_cfg, dict) else None

    # Build system prompt with avatar controls and tool information
    base_system_prompt = config.assistant["system_prompt"]
    system_prompt_with_avatar = build_prompt_with_avatar_controls(
        base_system_prompt,
        gesture_catalog=gesture_catalog,
        allowed_expressions=allowed_expressions,
    )
    system_prompt_with_tools = build_prompt_with_tools(
        system_prompt_with_avatar,
        tool_executor=tool_executor,
    )

    context_builder = ContextBuilder(
        system_prompt=system_prompt_with_tools,
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
        summary_store=summary_store,
        tool_executor=tool_executor,
        memory_retriever=memory_retriever,
        turn_finalizer=turn_finalizer,
        gesture_catalog=gesture_catalog,
    )

    logger.info(
        "Orchestrator built successfully (tools=%d)",
        len(tools),
    )

    return orchestrator
