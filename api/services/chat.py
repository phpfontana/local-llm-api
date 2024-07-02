from typing import List, Tuple
from fastapi import HTTPException
from api.schemas.chat import ChatHistory
from langchain_core.language_models.llms import BaseLLM
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def format_chat_history(history:List[ChatHistory]) -> List[Tuple[str, str]]:
    """
    Format chat history for use in the chat pipeline.

    Args:
        history (List[ChatHistory]): The chat history.

    Returns:
        List[Tuple[str, str]]: The formatted chat history.
    """
    return [(entry.role, entry.content) for entry in history]


def format_chat_template(messages:List[Tuple[str, str]], system_prompt:str, user_prompt:str="{input}"):
    """

    Args:
        messages (List[Tuple[str, str]]): The chat messages.
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.

    Returns:
        ChatPromptTemplate: The formatted chat template.
    """
    try:
        messages = [("system", system_prompt)] + messages + [("human", user_prompt)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatPromptTemplate.from_messages(messages)


def load_chat_ollama(model_name:str='llama3:instruct', base_url:str=None, **kwargs) -> BaseLLM:
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
    try:
        llm = ChatOllama(model=model_name, base_url=base_url, streaming=True, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return llm