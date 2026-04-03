from enum import Enum


class AssistantState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    DREAMING = "dreaming"
    SEARCHING = "searching"
    RESPONDING = "responding"
