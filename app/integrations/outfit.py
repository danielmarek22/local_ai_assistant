from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock

from app.integrations.contracts import (
    AvatarOutfitEffect,
    CapabilityId,
    ContextContribution,
    InvocationContext,
    RegisteredTool,
    ToolResult,
    ToolSpec,
)


@dataclass
class AvatarWardrobe:
    catalog: dict[str, str]
    current_outfit: str
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        self.catalog = dict(self.catalog)
        if self.catalog and self.current_outfit not in self.catalog:
            raise ValueError(f"Unknown initial outfit: {self.current_outfit!r}")

    def select(self, outfit: str) -> tuple[bool, str | None]:
        with self._lock:
            url = self.catalog.get(outfit)
            if url is None:
                return False, None
            changed = outfit != self.current_outfit
            self.current_outfit = outfit
            return changed, url


class OutfitIntegration:
    name = "outfit"

    def __init__(self, wardrobe: AvatarWardrobe):
        self.wardrobe = wardrobe

    def registered_tools(self) -> list[RegisteredTool]:
        outfits = sorted(self.wardrobe.catalog)
        if not outfits:
            return []
        return [RegisteredTool(
            spec=ToolSpec(
                capability=CapabilityId("outfit", "change"),
                description=(
                    "Change Astra's outfit. Use when the user requests it or when a change "
                    "clearly fits the conversation; avoid changing outfits arbitrarily."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"outfit": {"type": "string", "enum": outfits}},
                    "required": ["outfit"],
                    "additionalProperties": False,
                },
            ),
            handler=self._change,
        )]

    def _change(
        self,
        arguments: Mapping[str, object],
        _context: InvocationContext,
    ) -> ToolResult:
        outfit = str(arguments["outfit"])
        changed, url = self.wardrobe.select(outfit)
        if url is None:
            return ToolResult.error(f"Outfit is unavailable: {outfit}")
        if not changed:
            return ToolResult.success(f"Astra is already wearing {outfit}.")
        return ToolResult.success(
            f"Changed Astra's outfit to {outfit}.",
            effects=(AvatarOutfitEffect(outfit=outfit, url=url),),
        )

    def context(self, _invocation: InvocationContext) -> ContextContribution | None:
        if not self.wardrobe.catalog:
            return None
        return ContextContribution(
            source=self.name,
            content=f"Astra's current outfit is {self.wardrobe.current_outfit}.",
        )
