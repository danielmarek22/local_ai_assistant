from pydantic import BaseModel


class AssistantSpeechEvent(BaseModel):
    text: str
    is_final: bool = False


class AssistantStateEvent(BaseModel):
    state: str
