from typing import Optional, List
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    model: str = "llama3:instruct"
    prompt: Optional[str] = None
    images: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    prompt: str
    response: str
