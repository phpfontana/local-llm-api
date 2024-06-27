from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from api.schemas import *
from api.config import OLLAMA_HOST, OLLAMA_PORT
from api.services.llms import load_llm_ollama
from api.services.document_loaders import *
from api.services.vectorstore import *
from api.services.text_splitters import *
from api.services.embeddings import *

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


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
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        # Instantiate RAG chain
        qa_chain = create_stuff_documents_chain(
            llm, rag_prompt
        )

        rag_chain = create_retrieval_chain(
            retriever, qa_chain
        )

        # Invoke RAG chain
        response = rag_chain.invoke(
            {"input": prompt}
        )

        # TODO: Add db support

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response

