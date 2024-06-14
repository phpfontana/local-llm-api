from fastapi import APIRouter, Depends, HTTPException
from app.schemas import ModelRequest
from app.utils import *

# Instantiate router
router = APIRouter(
    prefix="/api/rag", tags=["rag"], responses={404: {"description": "Not found"}},
)

@router.post("/")
async def main(request: ModelRequest):
    """
    Perform retrieval augmented generation.
    """
    # load embeddings model
    embeddings_model = load_embeddings_model(request.model)

    # load documents
    documents = load_documents(request.prompt)

    # split documents
    splits = split_documents(documents)

    

    pass
