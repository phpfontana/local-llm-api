from typing import Any
from fastapi import HTTPException
from langchain_core.language_models.llms import BaseLLM
from langchain_community.llms.ollama import Ollama
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoProcessor, AutoModelForPreTraining
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from typing import List, Tuple
from api.schemas.chat import *

def load_chat_llm(model_name: str, **kwargs):
    
    llm = ChatLlamaCpp(model_path=model_name, **kwargs)

    return llm

def format_chat_message(messages:List[Message]) -> List[Tuple[str, str]]:
    """
    Format chat history for use in the chat pipeline.

    Args:
        history (List[ChatHistory]): The chat history.

    Returns:
        List[Tuple[str, str]]: The formatted chat history.
    """
    return [(entry.role, entry.content) for entry in messages]

def generate_reponse(prompt, llm):
    return llm.invoke(prompt)