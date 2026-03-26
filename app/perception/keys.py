from enum import Enum

class PerceptionKey(str, Enum):
    """Explicit contract for all keys stored in the PerceptionState."""
    USER_INPUT = "user.input"
    MEMORY_RETRIEVED = "memory.retrieved"
    # Future keys can go here:
    # SYSTEM_TIME = "system.time"
    # VISION_SCENE = "vision.scene"