from typing import Any
from langchain_core.language_models.llms import BaseLLM


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
        for chunks in llm.stream(prompt):
            yield chunks
    except Exception as e:
        raise ValueError(f"Error generating: {str(e)}") from e