import json
import logging
from typing import Any, Optional
from pydantic import BaseModel, ValidationError
from app.logging import trace_event

logger = logging.getLogger("memory_reflector")

class NewMemorySpec(BaseModel):
    content: str
    category: str = "general"
    importance: int = 2

class ReflectorOutput(BaseModel):
    delete_ids: list[str]
    keep_ids: list[str]
    new_memories: list[NewMemorySpec]

class MemoryReflector:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def reflect_and_prune(self, days_old: int = 14) -> dict[str, Any]:
        """Manually triggered dream state to consolidate old memories."""
        if days_old < 0:
            raise ValueError("days_old must be >= 0")

        logger.info("Initiating Memory Reflection (Dream State)...")

        stale_memories = self.memory_store.get_stale(days_old)
        trace_event(
            "memory_reflector",
            "reflection_input",
            payload={
                "days_old": days_old,
                "stale_count": len(stale_memories),
                "stale_memories": stale_memories,
            },
        )

        result: dict[str, Any] = {
            "success": True,
            "days_old": days_old,
            "stale_count": len(stale_memories),
            "deleted_count": 0,
            "kept_count": 0,
            "created_count": 0,
            "delete_ids": [],
            "keep_ids": [],
            "new_memories": [],
            "error": None,
        }

        if not stale_memories:
            logger.info("No stale memories found older than %d days.", days_old)
            trace_event(
                "memory_reflector",
                "reflection_complete",
                payload={
                    "days_old": days_old,
                    "stale_count": 0,
                    "result": result,
                },
            )
            return result

        memory_text = self._format_for_prompt(stale_memories)
        stale_ids = {memory["id"] for memory in stale_memories}

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a memory consolidation AI. Review the following stale memories "
                    "that have not been accessed recently.\n\n"
                    "Your task:\n"
                    "1. Delete highly specific, obsolete, or duplicate facts.\n"
                    "2. Keep enduring, important facts.\n"
                    "3. If several deleted facts point to a broader pattern, create a 'new_memory' summarizing them.\n\n"
                    "Output ONLY valid JSON matching this schema:\n"
                    "{\n"
                    '  "delete_ids": ["id1", "id2"],\n'
                    '  "keep_ids": ["id3"],\n'
                    '  "new_memories": [{"content": "User prefers concise Python.", "category": "preference", "importance": 3}]\n'
                    "}"
                )
            },
            {
                "role": "user",
                "content": f"Stale Memories to Review:\n{memory_text}"
            }
        ]

        # Since this is an offline/manual task, we can afford a longer LLM generation time
        logger.info("Asking LLM to evaluate %d stale memories...", len(stale_memories))
        response = self.llm.chat(prompt, False)
        trace_event(
            "memory_reflector",
            "llm_output",
            payload={
                "days_old": days_old,
                "stale_count": len(stale_memories),
                "response": response,
            },
        )

        try:
            data = self._extract_json(response)
            if data is None:
                raise ValueError("No JSON object found in reflection response")

            parsed = ReflectorOutput.model_validate(data)
            delete_ids = [memory_id for memory_id in parsed.delete_ids if memory_id in stale_ids]
            ignored_delete_ids = [memory_id for memory_id in parsed.delete_ids if memory_id not in stale_ids]
            keep_ids = [memory_id for memory_id in parsed.keep_ids if memory_id in stale_ids]

            # Execute Deletions
            if delete_ids:
                self.memory_store.delete_memories(delete_ids)

            if ignored_delete_ids:
                logger.warning(
                    "Ignoring %d non-stale memory IDs returned for deletion",
                    len(ignored_delete_ids),
                )

            # Execute Additions
            for new_mem in parsed.new_memories:
                self.memory_store.add(
                    content=new_mem.content,
                    category=new_mem.category,
                    importance=new_mem.importance
                )

            result["delete_ids"] = delete_ids
            result["keep_ids"] = keep_ids
            result["new_memories"] = [memory.model_dump() for memory in parsed.new_memories]
            result["deleted_count"] = len(delete_ids)
            result["kept_count"] = len(keep_ids)
            result["created_count"] = len(parsed.new_memories)
            if ignored_delete_ids:
                result["ignored_delete_ids"] = ignored_delete_ids

            logger.info(
                "Reflection Complete: Deleted %d, Kept %d, Created %d new consolidated memories.", 
                len(delete_ids), len(keep_ids), len(parsed.new_memories)
            )
            trace_event(
                "memory_reflector",
                "reflection_complete",
                payload={
                    "days_old": days_old,
                    "stale_count": len(stale_memories),
                    "result": result,
                },
            )

        except (ValidationError, ValueError, TypeError) as e:
            logger.error("Failed to parse reflection output: %s\nRaw output: %s", e, response)
            result["success"] = False
            result["error"] = str(e)
            trace_event(
                "memory_reflector",
                "reflection_parse_error",
                payload={
                    "days_old": days_old,
                    "stale_count": len(stale_memories),
                    "error": str(e),
                    "response": response,
                },
            )

        return result

    def _format_for_prompt(self, memories: list[dict]) -> str:
        lines = []
        for m in memories:
            lines.append(f"ID: {m['id']} | Importance: {m['importance']} | Content: {m['content']}")
        return "\n".join(lines)

    def _extract_json(self, text: str) -> Optional[dict]:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{": continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
                if isinstance(parsed, dict): return parsed
            except json.JSONDecodeError: continue
        return None
