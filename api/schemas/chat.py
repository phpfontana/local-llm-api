from typing import Optional, List
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str  
    messages: List[Message]
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    message: Message
    done: bool
