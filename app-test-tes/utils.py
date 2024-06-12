from typing import List, Dict, Any, Tuple, Union, Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader, UnstructuredMarkdownLoader


def load_embeddings_model(model_name: str, model_kwargs: Dict[str, Any], encode_kwargs: Dict[str, Any]) -> Embeddings:
    """
    Get embeddings model.

    Args:
        model_name (str): Model name
        model_kwargs (Dict[str, Any]): Model kwargs
        encode_kwargs (Dict[str, Any]): Encode kwargs
    
    Returns:
        Embeddings: Embeddings model
    """

    # Instantiate embeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embeddings_model

def load_documents(path: str) -> List[Document]:
    """
    Load documents from directory.

    Args:
        path (str): Directory path
    
    Returns:
        List[Document]: List of Documents.
    """

    # Instantiate document loader
    loader = DirectoryLoader(
        path= path,
        use_multithreading=True,
        show_progress=True,
        silent_errors=True
    )

    # Load documents
    documents = loader.load()

    return documents


def split_documents(documents: List[Document], chunk_size:int=1000, chunk_overlap:int=200) -> List[Document]:
    """
    Split documents into chunks.

    Args:
        documents (List[Document]): List of documents
        chunk_size (int): Size of the chunk
        chunk_overlap (float): Overlap between chunks
    
    Returns:
        List[Document]: List of documents with chunks.
    """

    # Instantiate text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

    # Split text
    splits = text_splitter.split_documents(documents)

    return splits

def load_to_vectorstore(documents: List[Document], vector_db: VectorStore, embeddings_model: Embeddings, connection_args: Optional[Dict[str, Any]] = None) -> VectorStore:
    """
    Generate vectorstore from documents.

    Args:
        documents (List[Document]): List of documents
        vector_db (VectorStore): 
        embeddings_model: Embeddings model
    
    Returns:
        VectorStore: VectorStore
    """

    # Instantiate vectorstore
    vectorstore = vector_db.from_documents(
        documents=documents, 
        embeddings_model=embeddings_model,
        collection_name=connection_args.get("collection"),
        connection_args={'uri': connection_args.get("uri")}
    )
    
    return vectorstore