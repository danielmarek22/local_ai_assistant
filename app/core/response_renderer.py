"""Per-response avatar/text rendering state shared by generation paths."""

import logging
from typing import Iterator

from app.avatar.stream_processor import StreamProcessor
from app.core.events import AssistantSpeechEvent, AvatarAnimationEvent, AvatarExpressionEvent


logger = logging.getLogger("orchestrator")
_DEFAULT_AVATAR_EXPRESSION = "neutral"
RenderEvent = AssistantSpeechEvent | AvatarAnimationEvent | AvatarExpressionEvent


class ResponseRenderer:
    def __init__(self, session_id: str, *, allowed_animations=None, allowed_expressions=None):
        self.session_id = session_id
        self.text = ""
        self._expression_initialized = False
        self._processor = StreamProcessor(
            allowed_animations=allowed_animations,
            allowed_expressions=allowed_expressions,
        )

    def push(self, text: str) -> Iterator[RenderEvent]:
        yield from self._emit(self._processor.push(text))

    def finish(self) -> Iterator[RenderEvent]:
        yield from self._emit(self._processor.flush())
        if not self._expression_initialized:
            yield self._expression(_DEFAULT_AVATAR_EXPRESSION)

    def _expression(self, value: str) -> AvatarExpressionEvent:
        self._expression_initialized = True
        logger.info("[%s] Model selected avatar expression '%s'", self.session_id, value)
        return AvatarExpressionEvent(expression=value)

    def _emit(self, events: list[tuple[str, str]]) -> Iterator[RenderEvent]:
        for event_type, value in events:
            if event_type == "expression":
                yield self._expression(value)
                continue
            if event_type == "animation":
                logger.info("[%s] Model selected avatar animation '%s'", self.session_id, value)
                yield AvatarAnimationEvent(animation=value)
                continue
            if not value or not value.strip():
                continue
            if not self._expression_initialized:
                yield self._expression(_DEFAULT_AVATAR_EXPRESSION)
            self.text += value
            yield AssistantSpeechEvent(text=value)
