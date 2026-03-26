from dataclasses import dataclass
from typing import Optional
from enum import Enum

class ActionType(str, Enum):
    WEB_SEARCH = "web_search"
    WRITE_MEMORY = "write_memory"
    RESPOND = "respond"

@dataclass
class Action:
    type: ActionType
    payload: Optional[dict] = None