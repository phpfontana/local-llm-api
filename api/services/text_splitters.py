from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Any


def split_markdown_text(text: str, headers_to_split_on: List[Tuple[str, str]], **kwargs: Dict[str, Any]) -> List[Document]:
    """
    Split markdown document into sections.

    Args:
        document (List[Document]): List of documents
        headers_to_split_on (List[Tuple[str, str]]): Headers to split on
        **kwargs: Keyword arguments

    Returns:
        List[Document]: List of documents

    Raises:
        Exception: If failed to split markdown document
    """
    try:
        # Instantiate markdown splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, **kwargs
            )
        # Split text
        splits = markdown_splitter.split_text(text)
    except Exception as e:
        raise Exception(f"Failed to split markdown document: {e}")

    return splits

def split_text(text: List[str], chunk_size:int=1000, chunk_overlap:int=100, **kwargs: Dict[str, Any]) -> List[Document]:
    """
    Split text into sections.

    Args:
        text (str): Text
        **kwargs: Keyword arguments

    Returns:
        List[Document]: List of documents

    Raises:
        Exception: If failed to split text
    """
    try:
        # Instantiate text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
        )
        # Split text
        splits = text_splitter.create_documents(text)
    except Exception as e:
        raise Exception(f"Failed to split text: {e}")

    return splits