from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from typing import Optional, List

def load_embeddings_model_hf(model_name: Optional[str]="sentence-transformers/all-MiniLM-L6-v2") -> Embeddings:
    """
    load embeddings model.

    Args:
        model_name (str): Model name
    
    Returns:
        Embeddings: Embeddings model
    """
    try:
        # Instantiate embeddings
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=True)
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
        # Embed documents
        embeddings = embeddings_model.embed_documents(documents)
    except Exception as e:
        raise Exception(f"Failed to generate document embeddings: {e}")
    return embeddings

def generate_query_embeddings(query: str, embeddings_model: Embeddings) -> List[float]:
    """
    Embed query.

    Args:
        query (str): Query
        embeddings_model (Embeddings): Embeddings model
    
    Returns:
        List[float]: Embedding
    """
    try:
        # Embed query
        embedding = embeddings_model.embed_query(query)
    except Exception as e:
        raise Exception(f"Failed to generate query embeddings: {e}")    

    return embedding
