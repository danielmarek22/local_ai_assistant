from __future__ import annotations

import json


class BeliefSnapshotFormatter:
    def __init__(self, *, max_chars: int = 2000):
        self.max_chars = max(0, int(max_chars))

    def format(self, beliefs) -> str | None:
        lines: list[str] = []
        for belief in beliefs:
            value = json.dumps(belief.value, ensure_ascii=True, sort_keys=True)
            visibility = belief.visibility.value.lower()
            expiry = belief.expires_at.isoformat() if belief.expires_at else "until revised"
            lines.append(
                f"- {belief.subject}.{belief.predicate} = {value} "
                f"({visibility}; {expiry})"
            )
        if not lines or self.max_chars <= 0:
            return None
        for keep_count in range(len(lines), -1, -1):
            output_lines = lines[:keep_count]
            omitted = len(lines) - keep_count
            if omitted:
                output_lines.append(f"[+{omitted} belief record(s) omitted]")
            content = "\n".join(output_lines)
            if len(content) <= self.max_chars:
                return content
        return None


class BeliefContextProvider:
    def __init__(self, owner_agent_id: str, snapshot_service, formatter):
        self.owner_agent_id = owner_agent_id
        self.snapshot_service = snapshot_service
        self.formatter = formatter

    def context_for_turn(self, session_id: str) -> str | None:
        beliefs = self.snapshot_service.active_for_turn(
            self.owner_agent_id,
            session_id,
        )
        return self.formatter.format(beliefs)
