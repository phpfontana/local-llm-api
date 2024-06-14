from fastapi import APIRouter, Depends, HTTPException
from app.schemas import ModelRequest
from pymilvus import Milvus
from app.utils import *

MILVUS_URI = "http//milvus:19530"

# Instantiate router
router = APIRouter(
    prefix="/api/vectorstore", tags=["vectorstore"], responses={404: {"description": "Not found"}},
)

@router.post("/store")
async def post_store_vectors(request: ModelRequest):
    """
    Store vectors in Milvus.
    """
    # load embeddings model
    embeddings_model = load_embeddings_model(request.model)

    # load documents
    documents = load_documents(request.prompt)

    # split documents
    splits = split_documents(documents)

    # load to vectorestore
    vector_db = load_to_vectorstore(
        documents=splits, vector_db=VectorStore(), embeddings_model=embeddings_model,
        connection_args={"uri": MILVUS_URI, "collection": ""}
    )