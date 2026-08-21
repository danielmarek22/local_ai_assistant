from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CompletedUserTurn:
    owner_agent_id: str
    session_id: str
    user_message_id: int
    user_text: str
    observed_at: datetime
    timezone_name: str
