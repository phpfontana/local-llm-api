import os
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.language_models.llms import BaseLLM
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline

def load_llm(model_id: str, model_kwargs: Dict[str, Any], pipeline_kwargs: Dict[str, Any], task: str) -> BaseLLM:
    """
    Load language model.

    Args:
        model_name (str): Model name
        model_kwargs (Dict[str, Any]): Model kwargs
        pipeline_kwargs (Dict[str, Any]): Pipeline kwargs
    """
    
    # Load llm
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task=task,
        device=0,
        pipeline_kwargs=pipeline_kwargs,
        model_kwargs=model_kwargs
        )

    # Wrap over tokenizer template
    llm_engine_hf = ChatHuggingFace(llm=llm)

    return llm_engine_hf

def get_embeddings_model(model_name: str, model_kwargs: Dict[str, Any], encode_kwargs: Dict[str, Any]) -> Embeddings:
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

def get_embeddings(docs: List[str], embeddings_model: Embeddings) -> List[List[float]]:
    """
    Generate embeddings using embeddings model.

    Args:
        docs (List[str]): List of documents to embed
        embeddings_model: Embeddings model
    
    Returns:
        List[List[float]]: Array of embeddings
    """

    # Embed documents
    embeddings = embeddings_model.embed_documents(docs)

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

def get_vectorstore(documents: List[Document], embeddings_model: Embeddings):
    """
    Generate vectorstore from documents.

    Args:
        documents (List[Document]): List of documents
        embeddings_model: Embeddings model
    
    Returns:
        Chroma: Vectorstore
    """

    # Instantiate vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model
    )

    return vectorstore


def main():

    # # Load environment variables
    # load_dotenv('.env', override=True)

    # # Environment variables
    # NEO4J_URI = os.getenv("NEO4J_URI")
    # NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    # NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    # NEO4JDATABASE = os.getenv("NEO4J_DATABASE")
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # ARGS
    DIRECTORY_PATH = "./data/"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
    EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': False}

    LLM_MODEL = ""

    # Instantiate embeddings
    embeddings_model = get_embeddings_model(EMBEDDING_MODEL, EMBEDDING_MODEL_KWARGS, EMBEDDING_ENCODE_KWARGS)

    # Load documents
    documents = load_documents(DIRECTORY_PATH)

    # Split documents
    splits = split_documents(documents, chunk_size=1024, chunk_overlap=200)
    
    # Instantiate vectorstore
    vectorstore = get_vectorstore(splits, embeddings_model)

    # Instantiate retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )

    # Load llm
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="summarization",
        device=0,
        pipeline_kwargs={
            "max_new_tokens": 100,
            "top_k": 50,
            "temperature": 0.1,
            },
        )

    # Wrap over tokenizer template
    llm_engine_hf = ChatHuggingFace(llm=llm)

    # Instantiate prompt template
    template = """
    You are an expert at creating questions based the context materials and documentation.
    Your goal is to prepare a person for their exams and tests based on the context.
    You do this by asking questions about the context below:

    ------------
    {context}
    ------------

    Create questions that will prepare the coders or programmers for their tests.
    Make sure not to lose any important information. 
    Provide answers for the questions you create.
    the answers should be based on the context above.

    QUESTIONS:
    a. {question_1}
    b. {question_2}
    c. {question_3}
    d. {question_4}

    ANSWERS:
    
    """

    prompt = PromptTemplate.from_template(template)


    

if __name__ == '__main__':
    main()


# https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7