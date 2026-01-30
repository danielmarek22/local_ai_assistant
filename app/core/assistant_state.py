from enum import Enum


class AssistantState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    SEARCHING = "searching"
    RESPONDING = "responding"
