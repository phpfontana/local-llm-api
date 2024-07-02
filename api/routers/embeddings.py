from fastapi import APIRouter, HTTPException
from api.schemas.embeddings import *
from api.services.embeddings import *


# Instantiate router
router = APIRouter(
    prefix="/api/embeddings", tags=["embeddings"], responses={404: {"description": "Not found"}}
)


@router.post("/", response_model=EmbeddingsResponse, status_code=200)
async def main(request: EmbeddingsRequest):
    try:
        embeddings_model = load_embeddings_hf(request.model)
        embeddings = generate_embeddings(request.query, embeddings_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {e}")
    
    return EmbeddingsResponse(embeddings=embeddings)

