from __future__ import annotations

import json
from datetime import datetime

from pydantic import ValidationError

from app.beliefs.models import BeliefCandidateBatch
from app.beliefs.vocabulary import CANONICAL_PREDICATES


class BeliefExtractionError(RuntimeError):
    pass


class BeliefCandidateExtractor:
    collector_name = "submit_belief_candidates"

    def __init__(
        self,
        llm,
        *,
        max_candidates: int = 4,
        max_context_chars: int = 1000,
        max_context_messages: int = 2,
        max_tokens: int = 384,
        timeout_s: float = 30.0,
    ):
        self.llm = llm
        self.max_candidates = max(1, min(int(max_candidates), 8))
        self.max_context_chars = max(0, int(max_context_chars))
        self.max_context_messages = max(0, int(max_context_messages))
        self.max_tokens = max(64, int(max_tokens))
        self.timeout_s = max(1.0, float(timeout_s))

    def extract(
        self,
        *,
        user_text: str,
        disambiguating_context: list[dict],
        existing_beliefs: list,
        observed_at: datetime,
        timezone_name: str,
    ) -> BeliefCandidateBatch:
        if observed_at.tzinfo is None:
            raise BeliefExtractionError("Extractor observation clock must be timezone-aware")
        messages = self._messages(
            user_text,
            disambiguating_context,
            existing_beliefs,
            observed_at,
            timezone_name,
        )
        response = self.llm.chat(
            messages=messages,
            think_override=False,
            options_override={
                "temperature": 0.0,
                "num_predict": self.max_tokens,
            },
            timeout_override=self.timeout_s,
            tools=[self._collector_tool()],
        )
        try:
            calls = response.get("tool_calls", [])
            if not isinstance(calls, list) or len(calls) != 1:
                raise BeliefExtractionError("Extractor must return exactly one collector call")
            function = calls[0].get("function", {})
            if function.get("name") != self.collector_name:
                raise BeliefExtractionError("Extractor returned an unexpected collector call")
            arguments = function.get("arguments")
            if not isinstance(arguments, dict):
                raise BeliefExtractionError("Extractor collector arguments must be an object")
            batch = BeliefCandidateBatch.model_validate(arguments)
        except (AttributeError, TypeError, ValidationError) as exc:
            raise BeliefExtractionError(f"Malformed belief extractor output: {exc}") from exc
        if len(batch.operations) > self.max_candidates:
            raise BeliefExtractionError("Extractor returned too many candidate operations")
        return batch

    def _messages(
        self,
        user_text: str,
        context: list[dict],
        beliefs: list,
        observed_at: datetime,
        timezone_name: str,
    ) -> list[dict]:
        vocabulary = "\n".join(
            f"- {predicate}: {description}"
            for predicate, description in CANONICAL_PREDICATES.items()
        )
        belief_payload = [
            {
                "belief_id": belief.belief_id,
                "subject": belief.subject,
                "predicate": belief.predicate,
                "value": belief.value,
                "visibility": belief.visibility.value,
                "expires_at": belief.expires_at.isoformat() if belief.expires_at else None,
            }
            for belief in beliefs
        ]
        bounded_context = self._bounded_context(context)
        return [
            {
                "role": "system",
                "content": (
                    "Extract only explicit, temporary or revisable CURRENT information asserted "
                    "by the author of the current user message. The current user message is the "
                    "only evidence. Earlier conversation is disambiguating context only. Never "
                    "extract claims found only in quotations, code blocks, pasted conversations, "
                    "logs, tool output, examples, fiction, roleplay, meta-instructions, or statements "
                    "attributed to another person. Do not infer psychology, emotion, intent, or hidden "
                    "state. Stable facts and enduring preferences are memory-like, not beliefs. "
                    "Hypotheticals and future possibilities are not current beliefs.\n\n"
                    "Use AGENT_CURRENT for ordinary current location, activity, availability, temporary "
                    "physical condition, or world state. Use SESSION_CURRENT only when explicitly local "
                    "to this conversation/session. Use an existing belief ID and UPDATE when it represents "
                    "the same semantic property. Prefer this canonical vocabulary:\n"
                    f"{vocabulary}\n\n"
                    "Extend the vocabulary only when none fits, using 2-64 lowercase snake_case characters. "
                    "UPDATE preserves the target belief's visibility. Return at most the configured "
                    "number of operations through the collector tool.\n\n"
                    "Expiry policies:\n"
                    "- END_OF_SESSION: SESSION_CURRENT only; lasts until session deletion.\n"
                    "- AFTER_ONE_HOUR: exactly one hour after the observation clock.\n"
                    "- END_OF_LOCAL_DAY: midnight beginning the next local day.\n"
                    "- AFTER_TWENTY_FOUR_HOURS: exactly 24 hours after the observation clock.\n"
                    "- AFTER_SEVEN_DAYS: exactly seven days after the observation clock.\n"
                    "- UNTIL_EXPLICIT_DATETIME: a timezone-aware ISO value in explicit_until.\n"
                    "- NO_AUTOMATIC_EXPIRY: until replacement or invalidation.\n"
                    "Examples: 'for an hour' uses AFTER_ONE_HOUR; 'today' normally uses "
                    "END_OF_LOCAL_DAY; 'until Friday' uses UNTIL_EXPLICIT_DATETIME resolved "
                    "to the next Friday in the configured timezone from the supplied clock."
                ),
            },
            {
                "role": "user",
                "content": (
                    "AUTHORITATIVE OBSERVATION CLOCK:\n"
                    f"datetime={observed_at.isoformat()}\ntimezone={timezone_name}\n\n"
                    f"DISAMBIGUATING CONTEXT (not evidence):\n{bounded_context}\n\n"
                    "ALL VISIBLE UNDERLYING BELIEFS (for update resolution):\n"
                    f"{json.dumps(belief_payload, ensure_ascii=True, sort_keys=True)}\n\n"
                    "CURRENT USER MESSAGE (sole evidence):\n"
                    f"{user_text}"
                ),
            },
        ]

    def _bounded_context(self, context: list[dict]) -> str:
        if self.max_context_messages <= 0:
            return "[]"
        bounded = []
        for item in context[-self.max_context_messages:]:
            bounded.append({
                "role": str(item.get("role", "")),
                "content": str(item.get("content", ""))[: self.max_context_chars],
            })
        return json.dumps(bounded, ensure_ascii=True)

    def _collector_tool(self) -> dict:
        schema = BeliefCandidateBatch.model_json_schema()
        schema["properties"]["operations"]["maxItems"] = self.max_candidates
        return {
            "type": "function",
            "function": {
                "name": self.collector_name,
                "description": "Submit validated conversational belief candidate operations.",
                "parameters": schema,
            },
        }
