from dataclasses import dataclass
import logging

from app.logging import trace_event

logger = logging.getLogger("memory_retriever")


@dataclass(frozen=True)
class MemoryRetrievalResult:
    memory_context: str | None
    perception_value: str
    failed_sources: tuple[str, ...] = ()


class MemoryRetriever:
    def __init__(
        self,
        memory_store,
        history_store,
        semantic_limit: int = 3,
        episodic_limit: int = 3,
    ):
        self.memory = memory_store
        self.history = history_store
        self.semantic_limit = semantic_limit
        self.episodic_limit = episodic_limit

    def retrieve(self, query: str, session_id: str) -> MemoryRetrievalResult:
        logger.debug("[%s] Querying vector DB for memories", session_id)

        if not query:
            return MemoryRetrievalResult(
                memory_context=None,
                perception_value="No relevant past memories found.",
            )

        failed_sources = []
        try:
            semantic_memories = self.memory.get_relevant(query, limit=self.semantic_limit)
        except Exception:
            logger.exception("[%s] Semantic retrieval failed; continuing without it", session_id)
            semantic_memories = []
            failed_sources.append("semantic")

        try:
            episodic_memories = self.history.search_past_conversations(
                query,
                session_id,
                limit=self.episodic_limit,
            )
        except Exception:
            logger.exception("[%s] Episodic retrieval failed; continuing without it", session_id)
            episodic_memories = []
            failed_sources.append("episodic")

        memory_blocks = []
        if semantic_memories:
            memory_blocks.append(
                "Relevant Facts:\n" + "\n".join(f"- {memory}" for memory in semantic_memories)
            )
        if episodic_memories:
            memory_blocks.append(
                "Past Conversations:\n" + "\n".join(f"- {memory}" for memory in episodic_memories)
            )

        memory_context = "\n\n".join(memory_blocks) if memory_blocks else None
        trace_event(
            "memory_retriever",
            "retrieval_result",
            session_id=session_id,
            payload={
                "query": query,
                "semantic_memories": semantic_memories,
                "episodic_memories": episodic_memories,
                "memory_context": memory_context,
                "failed_sources": failed_sources,
            },
        )
        perception_value = f"\n{memory_context}\n" if memory_context else "No relevant past memories found."
        if failed_sources:
            diagnostic = f"Memory retrieval incomplete; unavailable sources: {', '.join(failed_sources)}."
            perception_value = f"{perception_value}\n{diagnostic}" if memory_context else diagnostic
        return MemoryRetrievalResult(
            memory_context=memory_context,
            perception_value=perception_value,
            failed_sources=tuple(failed_sources),
        )
