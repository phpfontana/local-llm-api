from fastapi import APIRouter, HTTPException
from api.config import MODELS_PATH
from api.schemas.embeddings import *
from api.services.embeddings import *

router = APIRouter(
    prefix="/api/embeddings", tags=["embeddings"], responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=EmbeddingsResponse, status_code=200)
async def main(request: EmbeddingsRequest):
    try:
        embeddings_model = load_embeddings(model_name=f"{MODELS_PATH}{request.model}")
        embeddings = generate_embeddings(request.prompt, embeddings_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {e}")
    
    return EmbeddingsResponse(embeddings=embeddings)