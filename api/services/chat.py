from typing import List, Tuple
from fastapi import HTTPException
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from api.schemas.chat import *

def load_chat_llm(model_name: str, **kwargs):

    try:
        llm = ChatLlamaCpp(model_path=model_name, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    return llm

def format_chat_message(messages:List[Message]) -> List[Tuple[str, str]]:
    return [(entry.role, entry.content) for entry in messages]

def generate_reponse(prompt, llm):
    return llm.invoke(prompt)

async def generate_streaming_response(prompt, llm):
    async for chunk in llm.astream(prompt):
        yield ChatResponse(message=Message(role="assistant", content=chunk.content), done=False)