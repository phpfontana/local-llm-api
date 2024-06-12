from typing import Optional
from pydantic import BaseModel
import torch

class EmbeddingRequest(BaseModel):
    model: Optional[str] = "sentence-transformers/all-mpnet-base-v2"
    prompt: str
    model_kwargs: Optional[dict] = {"device": "cpu" if not torch.cuda.is_available() else "cuda"}
    encode_kwargs: Optional[dict] = {"normalize_embeddings": False}

class RetrieveRequest(BaseModel):
    model: Optional[str] = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs: Optional[dict] = {"device": "cpu" if not torch.cuda.is_available() else "cuda"}
    encode_kwargs: Optional[dict] = {"normalize_embeddings": False}

    



