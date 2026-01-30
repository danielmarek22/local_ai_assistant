from pydantic import BaseModel

class AssistantSpeechEvent(BaseModel):
    text: str
    emotion: str = "neutral"
    is_final: bool = False
