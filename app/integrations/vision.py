from __future__ import annotations

from app.integrations.contracts import (
    EventId,
    EventSpec,
    NotificationPolicy,
    ReplayPolicy,
)


class VisionIntegration:
    name = "vision"

    def registered_tools(self):
        return []

    def registered_events(self) -> list[EventSpec]:
        return [
            EventSpec(
                event=EventId(self.name, "attention_detected"),
                description=(
                    "The opted-in local vision watchdog detected a significant screen or webcam event."
                ),
                payload_schema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["screen", "webcam"]},
                        "description": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "sha256": {"type": "string", "minLength": 1},
                    },
                    "required": ["source", "description", "sha256"],
                    "additionalProperties": False,
                },
                allowed_capabilities=(),
                notification_policy=NotificationPolicy.MODEL_DECIDES,
                replay_policy=ReplayPolicy.NEVER,
                priority=75,
                coalesce_window_s=5.0,
            )
        ]

    def context(self, _context):
        return None
