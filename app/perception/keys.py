from enum import Enum

class PerceptionKey(str, Enum):
    """Explicit contract for all keys stored in the PerceptionState."""
    USER_INPUT = "user.input"
    MEMORY_RETRIEVED = "memory.retrieved"
    SCREEN_SCENE = "vision.screen"
    WEBCAM_SCENE = "vision.webcam"
