from typing import Optional, List
from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    query: Optional[str] = None

class EmbeddingsResponse(BaseModel):
    embeddings: List[float]
