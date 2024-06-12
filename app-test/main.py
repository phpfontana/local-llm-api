from fastapi import FastAPI, HTTPException, Request
from typing import List
from app.utils import *
from app.models import EmbeddingRequest
import json

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings using embeddings model.

    Args:
        request (EmbeddingRequest): Request

    Returns:
        dict: A dictionary containing embeddings.
    """
    
    try:
        # Get embeddings model
        embeddings_model = get_embeddings_model(
            model_name=request.model, model_kwargs=request.model_kwargs, encode_kwargs=request.encode_kwargs
        )

        # Get embeddings
        embeddings = get_embeddings([request.prompt], embeddings_model)

        if not embeddings or not embeddings[0]:
            raise ValueError("Failed to generate embeddings")

        response = {
            "embeddings": embeddings[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return response

@app.get("/retrieve")
async def retrieve_documents(RetrieveRequest: RetrieveRequest):
    """
    Retrieve documents from vector store.

    """

    try:
        # Instantiate embeddings model
        embeddings_model = get_embeddings_model(
            model_name=RetrieveRequest.model, model_kwargs=RetrieveRequest.model_kwargs, encode_kwargs=RetrieveRequest.encode_kwargs
        )

        # Load documents
        documents = load_documents(RetrieveRequest.path)

        # Split documents
        splits = split_documents(
            documents, chunk_size=RetrieveRequest.chunk_size, chunk_overlap=RetrieveRequest.chunk_overlap
            )
        
        # Instantiate vectorstore
        vectorstore = get_vectorstore(splits, embeddings_model)

        # Instantiate retriever
        retriever = vectorstore.as_retriever(
            search_type=RetrieveRequest.search_type,
            search_kwargs=RetrieveRequest.search_kwargs
        )

        # Retrieve documents
        results = retriever.invoke(RetrieveRequest.prompt)

        response = {
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return response
        