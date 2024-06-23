from typing import Optional, List, Dict
from pydantic import BaseModel
from app.config import MILVUS_HOST, MILVUS_PORT
from langchain_core.documents import Document

class EmbeddingsRequest(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    query: Optional[str] = None

class EmbeddingsResponse(BaseModel):
    embeddings: List[float]

class GenerateRequest(BaseModel):
    model: str = "llama3"
    prompt: Optional[str] = None

class GenerateResponse(BaseModel):
    response: str

class VectorStoreRequest(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    prompt: Optional[str] = None
    collection_name: str = "default" 
    