import json

from app.core.session_ids import SessionDeletedError
from app.integrations.contracts import CapabilityId, RegisteredTool, ToolResult, ToolSpec
from app.scratchpad.context import bounded_entries
from app.scratchpad.store import (
    MAX_CONTENT_CHARS, MAX_QUERY_CHARS, MAX_RESULTS, ScratchpadConflict, ScratchpadNotFound,
)


class ScratchpadIntegration:
    name = "scratchpad"

    def __init__(self, store):
        self.store = store

    def registered_tools(self):
        content = {"type": "string", "minLength": 1, "maxLength": MAX_CONTENT_CHARS}
        identity = {"type": "string", "minLength": 1, "maxLength": 128}
        revision = {"type": "integer", "minimum": 1, "maximum": 2**63 - 2}
        limit = {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS}
        specs = {
            "write": ({"content": content}, ["content"],
                      "Persistent notes for information you consider useful later. Write a free-form note."),
            "read": ({"id": identity}, ["id"], "Read a complete note and its current revision."),
            "recent": ({"limit": limit, "offset": {"type": "integer", "minimum": 0, "maximum": 100000}}, [],
                       "Browse recent notes. Previews may be truncated; use read for full content."),
            "search": ({"query": {"type": "string", "minLength": 1, "maxLength": MAX_QUERY_CHARS},
                        "limit": limit}, ["query"], "Search notes semantically. Use read for full content."),
            "edit": ({"id": identity, "content": content, "expected_revision": revision},
                     ["id", "content", "expected_revision"],
                     "Replace a note using its current revision. On conflict read again before retrying."),
            "delete": ({"id": identity, "expected_revision": revision}, ["id", "expected_revision"],
                       "Delete a note using its current revision. On conflict read again before retrying."),
        }
        return [RegisteredTool(
            spec=ToolSpec(CapabilityId(self.name, action), description,
                          {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}),
            handler=lambda arguments, context, action=action: self._invoke(action, arguments, context),
        ) for action, (properties, required, description) in specs.items()]

    def _invoke(self, action, arguments, context):
        try:
            if action == "write":
                payload = self.store.write(arguments["content"], context.session_id)
            elif action == "read":
                payload = {"entry": self.store.read(arguments["id"])}
            elif action == "recent":
                payload = self.store.recent(arguments.get("limit", 10), arguments.get("offset", 0))
            elif action == "search":
                try:
                    entries = self.store.search(arguments["query"], arguments.get("limit", 5))
                except ValueError:
                    raise
                except Exception:
                    return ToolResult.unavailable("Notebook search is unavailable; read/recent still work.")
                payload = {"entries": entries, "available": True}
            elif action == "edit":
                payload = self.store.edit(arguments["id"], arguments["content"],
                                          arguments["expected_revision"], context.session_id)
            else:
                payload = self.store.delete(arguments["id"], arguments["expected_revision"], context.session_id)
        except SessionDeletedError:
            return ToolResult.denied("The originating conversation was deleted; note mutation rejected.")
        except ScratchpadConflict as exc:
            return ToolResult.error(str(exc), diagnostics={"category": "revision_conflict"})
        except ScratchpadNotFound as exc:
            return ToolResult.error(str(exc), diagnostics={"category": "not_found"})
        except ValueError as exc:
            return ToolResult.error(str(exc))
        if "entries" in payload:
            entries = payload["entries"]
            payload["entries"] = bounded_entries(entries, 10000)
            payload["returned_count"] = len(payload["entries"])
            payload["omitted_count"] = len(entries) - payload["returned_count"]
            if action == "recent":
                payload["next_offset"] = (
                    payload["offset"] + payload["returned_count"]
                    if payload["offset"] + payload["returned_count"] < payload["total"] else None
                )
        return ToolResult.success(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
