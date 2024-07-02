from typing import Optional, List
from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    query: Optional[str] = None

class EmbeddingsResponse(BaseModel):
    embeddings: List[float]

class GenerateRequest(BaseModel):
    model: str = "llama3"
    prompt: Optional[str] = None
    stream: bool = False

class GenerateResponse(BaseModel):
    response: str

class VectorStoreRequest(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    prompt: Optional[str] = None
    collection_name: str = "default" 