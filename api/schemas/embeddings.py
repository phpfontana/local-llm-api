from typing import Optional, List
from pydantic import BaseModel

class EmbeddingsRequest(BaseModel):
    model: str 
    prompt: Optional[str] = None

class EmbeddingsResponse(BaseModel):
    embeddings: List[float]
