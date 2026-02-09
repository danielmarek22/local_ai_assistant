from dataclasses import dataclass
from typing import List
from app.core.actions import Action


@dataclass
class Plan:
    actions: List[Action]
