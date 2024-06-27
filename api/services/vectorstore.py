from typing import Any, Dict, List
from langchain_core.vectorstores import VST
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


def load_vectorstore(vectorstore: VectorStore, documents: List[Document], embeddings_model: Embeddings, **kwargs: Dict[str, Any]) -> VST:
    """
    Loads documents into vectorstore.

    Args:
        vectorstore (VectorStore): Vector store
        documents (List[Document]): List of documents
        embeddings_model (Embeddings): Embeddings model
        **kwargs: Keyword arguments

    Returns:
        VST: Vector store
    """
    try:
        # Load vectorstore
        vectorstore = vectorstore.from_documents(
            documents=documents, embedding=embeddings_model, **kwargs
        )
    except Exception as e:
        raise Exception(f"Failed to load vectorstore: {e}")

    return vectorstore

def load_retriever(vectorstore: VectorStore, **kwargs: Dict[str, Any]) -> VectorStoreRetriever:
    """
    Loads retriever from vectorstore.

    Args:
        vectorstore (VectorStore): Vector store

    Returns:
        VectorStoreRetriever: Vector store retriever
    """
    try:
        # Load retriever
        retriever = vectorstore.as_retriever(**kwargs)
    except Exception as e:
        raise Exception(f"Failed to load retriever: {e}")

    return retriever

def retrieve_documents(retriever: VectorStoreRetriever, query: str, search_type: str, **kwargs: Dict[str, Any]) -> List[Document]:
    """
    Retrieve documents from vectorstore.

    Args:
        retriever (VectorStoreRetriever): Vector store retriever
        query (str): Query
        search_type (str): Search type
        **kwargs: Keyword arguments

    Returns:
        List[Document]: List of documents
    """
    try:
        # Retrieve documents
        documents = retriever(search_type=search_type, search_kwargs=kwargs).invoke(query)
    except Exception as e:
        raise Exception(f"Failed to retrieve documents: {e}")

    return documents

