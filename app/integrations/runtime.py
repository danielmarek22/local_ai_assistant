from __future__ import annotations

from collections.abc import Mapping

from app.integrations.contracts import (
    CapabilityId,
    InvocationContext,
    NotificationDelivery,
    NotificationRequest,
    RegisteredTool,
    ToolResult,
    ToolSpec,
)


class RuntimeIntegration:
    name = "runtime"
    notify_capability = CapabilityId("runtime", "notify")

    def registered_tools(self) -> list[RegisteredTool]:
        return [
            RegisteredTool(
                spec=ToolSpec(
                    capability=self.notify_capability,
                    description=(
                        "Notify the user about an autonomous event outcome. Use only when the "
                        "user benefits from an interruption; otherwise finish silently."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "minLength": 1, "maxLength": 4000},
                            "delivery": {"type": "string", "enum": ["text", "speech"]},
                        },
                        "required": ["message", "delivery"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._notify,
            )
        ]

    @staticmethod
    def _notify(arguments: Mapping[str, object], context: InvocationContext) -> ToolResult:
        if context.event_id is None or context.notification_callback is None:
            return ToolResult.denied("Notifications are only available during autonomous event turns.")
        request = NotificationRequest(
            message=str(arguments["message"]).strip(),
            delivery=NotificationDelivery(str(arguments["delivery"])),
        )
        if not context.notification_callback(request):
            return ToolResult.denied("This event does not allow a user notification.")
        return ToolResult.success("The notification has been queued for delivery after this event turn.")

    def context(self, _context: InvocationContext):
        return None
