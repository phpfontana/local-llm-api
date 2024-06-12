from typing import Optional, Dict, Any
from pydantic import BaseModel

class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None