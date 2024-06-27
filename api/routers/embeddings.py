from fastapi import APIRouter, HTTPException
from api.schemas import EmbeddingsRequest, EmbeddingsResponse
from api.services.embeddings import *


# Instantiate router
router = APIRouter(
    prefix="/api/embeddings", 
    tags=["embeddings"], 
    responses={404: {"description": "Not found"}}
)


@router.post("/", response_model=EmbeddingsResponse)
async def main(request: EmbeddingsRequest):
    """
    Generate embeddings for query.
    """
    # Extract request
    model = request.model
    query = request.query

    try:
        # Load embeddings model
        embeddings_model = load_embeddings_model_hf(model)

        # Generate embeddings
        embeddings = generate_query_embeddings(query, embeddings_model)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {e}")
    
    return {"embeddings": embeddings}


