from typing import Any
from langchain_core.language_models.llms import BaseLLM
from langchain_community.llms.ollama import Ollama


def load_llm_ollama(model_name:str='llama3:instruct', base_url:str=None, **kwargs) -> BaseLLM:
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
        llm = Ollama(model=model_name, base_url=base_url, **kwargs)
    except Exception as e:
        raise ValueError(f"Error loading model {model_name}: {e}") from e        
    
    return llm


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
        raise ValueError(f"Error generating: {str(e)}") from e


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
        async for chunk in llm.astream(prompt):
            content = chunk.replace("\n", "<br>")
            yield content
    except Exception as e:
        raise ValueError(f"Error generating: {str(e)}") from e