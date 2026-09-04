from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.integrations.contracts import CapabilityId, ToolSpec


def _object_schema(
    properties: Mapping[str, object],
    required: Sequence[str] = (),
) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def _player_name_schema() -> dict[str, object]:
    return {
        "type": "string",
        "description": "Exact Minecraft player name.",
        "pattern": "^[A-Za-z0-9_]+$",
        "minLength": 1,
        "maxLength": 16,
    }


def build_mindcraft_tool_specs(namespace: str = "mindcraft") -> tuple[ToolSpec, ...]:
    """Return the declarative capability contract for a Mindcraft integration."""
    return (
        ToolSpec(
            capability=CapabilityId(namespace, "stop"),
            description=(
                "Stop the configured Minecraft agent's current action and any "
                "continuous goal. Use this for an immediate stop request."
            ),
            input_schema=_object_schema({}),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "say"),
            description=(
                "Send an exact message through the configured bot's Minecraft chat "
                "without invoking Mindcraft's planning model."
            ),
            input_schema=_object_schema(
                {
                    "message": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                    },
                },
                ("message",),
            ),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "go_to_player"),
            description=(
                "Move the configured Minecraft agent near a player. Use this direct "
                "action instead of delegating the request to Mindcraft's model."
            ),
            input_schema=_object_schema(
                {
                    "player_name": _player_name_schema(),
                    "closeness": {
                        "type": "number",
                        "description": "Desired distance from the player in blocks.",
                        "minimum": 0,
                        "maximum": 128,
                    },
                },
                ("player_name", "closeness"),
            ),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "follow_player"),
            description=(
                "Continuously follow a player at a specified distance. Use "
                "mindcraft__stop to stop following."
            ),
            input_schema=_object_schema(
                {
                    "player_name": _player_name_schema(),
                    "follow_distance": {
                        "type": "number",
                        "description": "Following distance in blocks.",
                        "minimum": 0,
                        "maximum": 128,
                    },
                },
                ("player_name", "follow_distance"),
            ),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "collect_blocks"),
            description=(
                "Collect a bounded number of nearby blocks of one Minecraft block "
                "type without invoking Mindcraft's planning model."
            ),
            input_schema=_object_schema(
                {
                    "block_type": {
                        "type": "string",
                        "description": "Minecraft block identifier, such as oak_log.",
                        "pattern": "^[a-z0-9_]+$",
                        "minLength": 1,
                        "maxLength": 64,
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of blocks to collect.",
                        "minimum": 1,
                        "maximum": 256,
                    },
                },
                ("block_type", "count"),
            ),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "collect_resource"),
            description=(
                "Collect a Minecraft resource using semantic names such as coal, iron, "
                "diamond, or redstone. Mindcraft resolves the resource to valid block IDs."
            ),
            input_schema=_object_schema(
                {
                    "resource": {
                        "type": "string",
                        "pattern": "^[a-z0-9_]+$",
                        "minLength": 1,
                        "maxLength": 64,
                    },
                    "count": {"type": "integer", "minimum": 1, "maximum": 256},
                },
                ("resource", "count"),
            ),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "chop_tree"),
            description=(
                "Find the nearest tree log type and chop a bounded number of logs from it."
            ),
            input_schema=_object_schema(
                {"max_logs": {"type": "integer", "minimum": 1, "maximum": 64}},
                ("max_logs",),
            ),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "capture_view"),
            description=(
                "Capture the Minecraft agent's current first-person view for direct "
                "visual inspection by the assistant."
            ),
            input_schema=_object_schema({}),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "look_at_position"),
            description="Turn toward coordinates, capture the view, and inspect it.",
            input_schema=_object_schema(
                {
                    "x": {"type": "integer", "minimum": -30000000, "maximum": 30000000},
                    "y": {"type": "integer", "minimum": -64, "maximum": 320},
                    "z": {"type": "integer", "minimum": -30000000, "maximum": 30000000},
                },
                ("x", "y", "z"),
            ),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "look_at_player"),
            description="Look at a player or in the same direction as that player.",
            input_schema=_object_schema(
                {
                    "player_name": _player_name_schema(),
                    "direction": {"type": "string", "enum": ["at", "with"]},
                },
                ("player_name", "direction"),
            ),
        ),
        ToolSpec(
            capability=CapabilityId(namespace, "send_message"),
            description=(
                "Delegate a complex, open-ended objective or conversational message to "
                "the configured Mindcraft agent and its planning model. Prefer a typed "
                "Mindcraft action when one directly matches the request."
            ),
            input_schema=_object_schema(
                {
                    "message": {
                        "type": "string",
                        "description": "The instruction or message for the Minecraft agent.",
                        "minLength": 1,
                        "maxLength": 4000,
                    },
                },
                ("message",),
            ),
        ),
    )
