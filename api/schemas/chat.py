from typing import Optional, List
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str  
    messages: List[Message]
    stream: Optional[str] = False

class ChatResponse(BaseModel):
    model: str
    message: Message
    done: bool
