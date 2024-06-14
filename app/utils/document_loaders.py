from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List, Dict, Any


def load_markdown_document(file_path: str, **kwargs: Dict[str, Any]) -> List[Document]:
    """
    Loads markdown document.

    Args:
        file_path (str): File path
        **kwargs: Keyword arguments

    Returns:
        List[Document]: List of documents

    Raises:
        Exception: If failed to load markdown document
    """
    try:
        # Instantiate loader
        loader = UnstructuredMarkdownLoader(file_path=file_path, **kwargs)
        # Load document
        document = loader.load()
    except Exception as e:
        raise Exception(f"Failed to load markdown document: {e}")
    return document

def load_pdf_document(file_path: str, **kwargs: Dict[str, Any]) -> List[Document]:
    """
    Loads PDF document.

    Args:
        file_path (str): File path
        **kwargs: Keyword arguments

    Returns:
        List[Document]: List of documents

    Raises:
        Exception: If failed to load PDF document
    """
    try:
        # Instantiate loader
        loader = PyPDFLoader(file_path=file_path, **kwargs)
        # Load document
        document = loader.load()
    except Exception as e:
        raise Exception(f"Failed to load PDF document: {e}")
    return document
