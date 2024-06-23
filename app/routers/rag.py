from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.schemas import *
from app.config import OLLAMA_HOST, OLLAMA_PORT
from app.services.llms import load_llm_ollama
from app.services.document_loaders import *
from app.services.vectorstore import *
from app.services.text_splitters import *
from app.services.embeddings import *

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


# Instantiate router
router = APIRouter(
    prefix="/api/rag", tags=["rag"], responses={404: {"description": "Not found"}},
)

@router.post("/")
async def main(prompt: str, file: UploadFile = File(...)):
    try:
        # Read file contents
        file_name = file.filename
        contents =  await file.read()  

        # TODO: Add support for other file types

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        # Split document
        splits = split_markdown_text(
            text=contents.decode("utf-8"), headers_to_split_on=headers_to_split_on
        )

        # Load embeddings model
        embeddings_model = load_embeddings_model_hf()

        # Load vectorstore
        vectorstore = load_vectorstore(
            vectorstore=Chroma(), documents=splits, embeddings_model=embeddings_model
        )

        # Load retriever
        retriever = load_retriever(vectorstore)

        # Load LLM
        llm = load_llm_ollama(base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

        # Instantiate prompt template
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.

        {context}

        Question: {question}

        Helpful Answer:"""

        custom_rag_prompt = PromptTemplate.from_template(template)
        
        # Instantiate RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            |custom_rag_prompt
            |llm
        )

        # Invoke RAG chain
        response = rag_chain.invoke(prompt)

        # TODO: 

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response

