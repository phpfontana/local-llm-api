from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.language_models.llms import BaseLLM
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import DirectoryLoader


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

def load_llm_hf(model_id: str, task: str, pipeline_kwargs: Dict[str, Any]=None) -> BaseLLM:
    """
    Load language model.

    Args:
        model_name (str): Model name
        task (str): Task
        pipeline_kwargs (Dict[str, Any]): Pipeline kwargs
    """
 
    # Load model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Load pipeline
    pipe = pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs
    )

    # Instantiate LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


def load_llm_ollama(model_name: str, pipeline_kwargs: Optional[Dict[str, Any]]=None) -> BaseLLM:
    """
    Load large language model from Ollama.

    Args:
        model_name (str): The name of the model to load
        pipeline_kwargs Optional(Dict[str, Any]): The pipeline actions.

    Returns:
        BaseLLM: The loaded language model.
    
    Raises:
        ValueError: If there is an error loading the model
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    try:
        return Ollama(model=model_name, **pipeline_kwargs, base_url="http://ollama:11434")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading model: {str(e)}")
    

async def generate_response(prompt: str, llm: BaseLLM) -> Any:
    """
    Generate a response using large language model.

    Args:
        prompt (String): The user prompt.
        llm (BaseLLM): The loaded language model.

    Returns:
        Any: The generated response or a streaming response
    """
    try:
        return llm.invoke(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating: {str(e)}")


async def generate_streaming_response(prompt: str, llm: BaseLLM) -> Any:
    """
    Generate a response using large language model.

    Args:
        promt (String): The llm prompt.
        llm (BaseLLM): The loaded language model.

    Returns:
        Any: The generated streaming response.
    """
    try:
        for chunks in llm.astream(prompt):
            yield chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating: {str(e)}")
