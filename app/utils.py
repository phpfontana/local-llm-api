from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain_core.embeddings import Embeddings

def load_embeddings_model(model_name: Optional[str]) -> Embeddings:
    """
    load embeddings model.

    Args:
        model_name (str): Model name
    
    Returns:
        Embeddings: Embeddings model
    """

    # Instantiate embeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name        
    )

    return embeddings_model

def generate_embeddings(prompt: str, embeddings_model: Embeddings) -> List[List[float]]:
    """
    Generate embeddings using embeddings model.

    Args:
        docs (List[str]): List of documents to embed
        embeddings_model: Embeddings model
    
    Returns:
        List[List[float]]: Array of embeddings
    """

    # Embed documents
    embeddings = embeddings_model.embed_query(prompt)

    return embeddings
