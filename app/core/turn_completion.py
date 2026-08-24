from dataclasses import dataclass
from datetime import datetime

from app.core.conversation import InputSource, SenderType


@dataclass(frozen=True)
class CompletedUserTurn:
    owner_agent_id: str
    session_id: str
    user_message_id: int
    user_text: str
    observed_at: datetime
    timezone_name: str
    sender_id: str = "local-human"
    sender_display_name: str = "You"
    sender_type: SenderType = SenderType.HUMAN
    input_source: InputSource = InputSource.LOCAL_TEXT
