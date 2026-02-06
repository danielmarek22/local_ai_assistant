from dataclasses import dataclass
from typing import Optional


@dataclass
class Action:
    type: str
    payload: Optional[dict] = None