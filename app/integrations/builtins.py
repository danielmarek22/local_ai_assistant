from __future__ import annotations

from collections.abc import Mapping

from app.integrations.contracts import (
    ApprovalRequest,
    CapabilityId,
    InvocationContext,
    RegisteredTool,
    ToolResult,
    ToolSpec,
)
from app.services.memory_action_handler import MemoryActionHandler
from app.tools.bash_execution import BashExecutionTool
from app.tools.web_search import WebSearchTool


class WebIntegration:
    name = "web"

    def __init__(self, tool: WebSearchTool):
        self.tool = tool

    def registered_tools(self) -> list[RegisteredTool]:
        return [RegisteredTool(
            spec=ToolSpec(
                capability=CapabilityId(self.name, "search"),
                description=self.tool.description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The specific search query to execute",
                            "minLength": 1,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            handler=self._search,
            available=lambda: self.tool.is_available,
        )]

    def _search(
        self,
        arguments: Mapping[str, object],
        _context: InvocationContext,
    ) -> ToolResult:
        result = self.tool.run(str(arguments["query"]))
        if not result:
            return ToolResult.error("Web search returned no usable results.")
        return ToolResult.success(result)

    def context(self, _invocation: InvocationContext):
        return None


class MemoryIntegration:
    name = "memory"

    def __init__(self, memory_action_handler: MemoryActionHandler):
        self.memory_action_handler = memory_action_handler

    def registered_tools(self) -> list[RegisteredTool]:
        return [RegisteredTool(
            spec=ToolSpec(
                capability=CapabilityId(self.name, "write"),
                description=(
                    "Use only for durable events, decisions, instructions, experiences, or narrative "
                    "context worth recalling in a future conversation. Never use for current "
                    "activities, temporary states, ordinary factual claims, preferences, or "
                    "propositions that belong in beliefs__update. Do not duplicate the same "
                    "proposition through both tools."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The fact or instruction to persist in memory.",
                            "minLength": 1,
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional memory category, such as general or preference.",
                        },
                        "importance": {
                            "type": "integer",
                            "description": "Optional importance score used for ranking and freshness.",
                        },
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            ),
            handler=self._write,
        )]

    def _write(
        self,
        arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        payload = dict(arguments)
        applied = self.memory_action_handler.handle_payload(context.session_id, payload)
        content = str(payload.get("content", ""))
        if not applied:
            return ToolResult.error("Memory write was rejected by the memory policy.")
        return ToolResult.success(f"Memory write accepted: {content}")

    def context(self, _invocation: InvocationContext):
        return None


class ShellIntegration:
    name = "shell"

    def __init__(self, tool: BashExecutionTool):
        self.tool = tool

    def registered_tools(self) -> list[RegisteredTool]:
        return [RegisteredTool(
            spec=ToolSpec(
                capability=CapabilityId(self.name, "execute"),
                description=self.tool.description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute",
                            "minLength": 1,
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
            handler=self._execute,
        )]

    def _execute(
        self,
        arguments: Mapping[str, object],
        context: InvocationContext,
    ) -> ToolResult:
        command = str(arguments["command"])
        approval_callback = None
        if context.approval_callback is not None:
            def approval_callback(request: dict) -> bool:
                return context.approval_callback(ApprovalRequest(
                    capability=CapabilityId(self.name, "execute"),
                    title="Approve command?",
                    reason=request["reason"],
                    detail_label="Command",
                    detail=request["command"],
                ))

        output = self.tool.run(command, approval_callback=approval_callback)
        content = output or "The command returned no output."
        if content.startswith("Permission Denied:"):
            return ToolResult.denied(content)
        if content.startswith(("Command failed", "Command timed out", "System error")):
            return ToolResult.error(content)
        return ToolResult.success(content)

    def context(self, _invocation: InvocationContext):
        return None
