from typing import Optional, List
from pydantic import BaseModel


class QARequest(BaseModel):
    model: str = "llama3:instruct"
    prompt: Optional[str] = None

class QAResponse(BaseModel):
    prompt: str
    response: str
