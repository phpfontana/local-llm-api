from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from api.schemas import *
from api.config import OLLAMA_HOST, OLLAMA_PORT, RAG_TEMPLATE, CHROMA_PATH
from api.services.llms import *
from api.services.document_loaders import *
from api.services.text_splitters import *
from api.services.vectorstore import *
from api.services.embeddings import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.pydantic_v1 import BaseModel, Field

class QA(BaseModel):
    """A question and answer pair."""
    question: str = Field(description="The question about the context")
    answer: str = Field(description="The answer to the question")
    rating: Optional[int] = Field(description="The difficulty of the question")


# Instantiate router
router = APIRouter(
    prefix="/api/qa", tags=["qa"], responses={404: {"description": "Not found"}},
)



@router.post("/")
async def main(prompt:str, file: UploadFile = File(...)):
    try:
        # Read file contents
        file_name = file.filename
        contents = await file.read()

        # Split document
        splits = split_text([contents.decode('utf-8')])

        # Load embeddings model
        embeddings_model = load_embeddings_model_hf()
        
        # Load vectorstore
        vectorstore = load_chroma_vectorstore(
            documents=splits, embeddings_model=embeddings_model, 
        )

        # Load retriever
        retriever = load_retriever(vectorstore)

        # retrieve documents
        docs = retriever.invoke(prompt)

        # Use retrieved documents as context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Load LLM
        llm = OllamaFunctions(model="llama3:instruct", base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

        structured_llm = llm.with_structured_output(QA)
            
        # Invoke LLM
        response = structured_llm.invoke(f"Generate a question using the following context:\n\n{context}" ) 

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "input": prompt,
        "output": response,
    }
