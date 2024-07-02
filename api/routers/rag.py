from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from api.schemas import *
from api.config import OLLAMA_HOST, OLLAMA_PORT, RAG_TEMPLATE, CHROMA_PATH
from api.services.llms import load_llm_ollama
from api.services.document_loaders import *
from api.services.text_splitters import *
from api.services.vectorstore import *
from api.services.embeddings import *
from langchain_core.prompts import ChatPromptTemplate


# Instantiate router
router = APIRouter(
    prefix="/api/rag", tags=["rag"], responses={404: {"description": "Not found"}},
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

        # Generate prompt
        prompt_template = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        rag_prompt = prompt_template.format(context=context, question=prompt)

        # Load LLM
        llm = load_llm_ollama(base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")
            
        # Invoke LLM
        response = llm.invoke(rag_prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "input": prompt,
        "context": docs,
        "output": response,
    }

