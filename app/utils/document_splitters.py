from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def split_documents(documents: List[Document], chunk_size:int=1000, chunk_overlap:int=200, **kwargs) -> List[Document]:
    """
    Split documents into chunks.

    Args:
        documents (List[Document]): List of documents
        chunk_size (int): Size of the chunk
        chunk_overlap (float): Overlap between chunks
    
    Returns:
        List[Document]: List of documents with chunks.
    """
    try:
        # Instantiate text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True, **kwargs
            )
        # Split text
        splits = text_splitter.split_documents(documents)
    except Exception as e:
        raise Exception(f"Failed to split documents: {e}")

    return splits
