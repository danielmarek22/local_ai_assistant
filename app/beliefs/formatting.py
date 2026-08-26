from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class BeliefContextPreview:
    beliefs: tuple
    formatted_body: str | None


class BeliefSnapshotFormatter:
    def __init__(self, *, max_chars: int = 2000):
        self.max_chars = max(0, int(max_chars))

    def format(self, beliefs) -> str | None:
        lines: list[str] = []
        for belief in beliefs:
            value = json.dumps(belief.value, ensure_ascii=True, sort_keys=True)
            visibility = belief.visibility.value.lower()
            expiry = belief.expires_at.isoformat() if belief.expires_at else "until revised"
            subject_name = json.dumps(belief.subject_display_name, ensure_ascii=True)
            subject_id = json.dumps(belief.subject_id, ensure_ascii=True)
            source_name = json.dumps(belief.source_sender_display_name, ensure_ascii=True)
            source_id = json.dumps(belief.source_sender_id, ensure_ascii=True)
            if belief.epistemic_status.value == "SELF_REPORT":
                provenance = f"self-report by {source_name} source_id={source_id}"
            else:
                provenance = f"claim by {source_name} source_id={source_id}"
            lines.append(
                f"- subject={subject_name} id={subject_id}; {belief.predicate} "
                f"= {value} ({provenance}; {visibility}; {expiry}; "
                f"source message {belief.source_message_id})"
            )
        if not lines or self.max_chars <= 0:
            return None
        output_lines: list[str] = []
        omitted = 0
        used_chars = 0
        for line in lines:
            separator_chars = 1 if output_lines else 0
            if used_chars + separator_chars + len(line) <= self.max_chars:
                output_lines.append(line)
                used_chars += separator_chars + len(line)
            else:
                omitted += 1

        if omitted:
            footer = f"[+{omitted} belief(s) omitted]"
            separator_chars = 1 if output_lines else 0
            if used_chars + separator_chars + len(footer) <= self.max_chars:
                output_lines.append(footer)

        return "\n".join(output_lines) or None


class BeliefContextProvider:
    def __init__(self, owner_agent_id: str, snapshot_service, formatter):
        self.owner_agent_id = owner_agent_id
        self.snapshot_service = snapshot_service
        self.formatter = formatter

    def context_for_turn(self, session_id: str) -> str | None:
        return self.preview_for_turn(session_id).formatted_body

    def preview_for_turn(self, session_id: str) -> BeliefContextPreview:
        beliefs = self.snapshot_service.active_for_turn(
            self.owner_agent_id,
            session_id,
        )
        return BeliefContextPreview(
            beliefs=tuple(beliefs),
            formatted_body=self.formatter.format(beliefs),
        )
