from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List

def load_embeddings(model_name: str, **kwargs):
    """
    load embeddings model from HuggingFace.

    Args:
        model_name (str): Model name
    
    Returns:
        Embeddings: Embeddings model
    """
    try:
        embeddings_model = LlamaCppEmbeddings(model_path=model_name, **kwargs)
    except Exception as e:
        raise Exception(f"Failed to load embeddings model: {e}")
    
    return embeddings_model

def generate_document_embeddings(documents: List[str], embeddings_model: Embeddings) -> List[List[float]]:
    """
    Embed documents.

    Args:
        documents (List[str]): List of documents
        embeddings_model (Embeddings): Embeddings model
    
    Returns:
        List[List[float]]: List of embeddings
    """
    try:
        embeddings = embeddings_model.embed_documents(documents)
    except Exception as e:
        raise Exception(f"Failed to generate document embeddings: {e}")
    return embeddings

def generate_embeddings(query: str, embeddings_model: Embeddings) -> List[float]:
    """
    Embed query.

    Args:
        query (str): Query
        embeddings_model (Embeddings): Embeddings model
    
    Returns:
        List[float]: Embedding
    """
    try:
        embedding = embeddings_model.embed_query(query)
    except Exception as e:
        raise Exception(f"Failed to generate query embeddings: {e}")    

    return embedding