from typing import Optional, List
from pydantic import BaseModel


class ChatHistory(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "llama3:instruct"
    prompt: str
    messages: List[ChatHistory]


class ChatResponse(BaseModel):
    message: ChatHistory
    
