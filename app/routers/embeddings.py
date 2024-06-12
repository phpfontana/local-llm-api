from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from app.utils import generate_embeddings, load_embeddings_model
from app.schemas import EmbeddingsRequest

# Instantiate router
router = APIRouter(
    prefix="/api/embeddings", tags=["embeddings"], responses={404: {"description": "Not found"}}
)

@router.put("/")
async def put_embeddings(request: EmbeddingsRequest):
    
    models = {
        "all-mpnet": "sentence-transformers/all-mpnet-base-v2",
        "all-minilm": "sentence-transformers/all-MiniLM-L6-v2"
    }

    if request.model in models:
        # Load embeddings model
        embeddings_model = load_embeddings_model(models[request.model])
    
    else:
        try:
            # Load embeddings model from user input
            embeddings_model = load_embeddings_model(request.model)
        except:
            pass
    
    # Generate embeddings
    embeddings = generate_embeddings(
        prompt=request.prompt, embeddings_model=embeddings_model)

    response = {
        "embeddings": embeddings,
        
    }

    return response




