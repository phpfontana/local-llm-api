from fastapi import APIRouter, HTTPException
from api.config import OLLAMA_URL
from api.schemas.chat import *
from api.services.chat import *

from langchain_core.output_parsers import StrOutputParser


# Instantiate router
router = APIRouter(
    prefix="/api/chat", tags=["chat"], responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=ChatResponse, status_code=200)
async def main(request: ChatRequest):
    try:
        history = request.messages
        
        chat_history = format_chat_history(history)
        
        system_prompt = "You are a chat assistant called LoLLa3, Local LLama3 Assistant. use the chat history to generate a response."
        
        
        chat_template = format_chat_template(chat_history, system_prompt)

        parser = StrOutputParser()

        llm = load_chat_ollama(request.model, base_url=OLLAMA_URL)
        
        chain = chat_template | llm | parser

        response = chain.invoke(
            {
                "input": request.prompt
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return ChatResponse(
        message=ChatHistory(role="ai", content=response),
    )
