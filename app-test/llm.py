import torch
from typing import Dict, Any, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

def load_huggingface_llm(model_id: str, task: str, device: int, model_kwargs: Optional[Dict[str, Any]] = None, pipeline_kwargs: Optional[Dict[str, Any]] = None) -> BaseLLM:
    """
    Load language model from HuggingFace.

    Args:
        model_id (str): Model id
        model_kwargs (Dict[str, Any]): Model kwargs
        pipeline_kwargs (Dict[str, Any]): Pipeline kwargs
        task (str): Task
    """
    
    # Load llm
    pipe = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task=task,
        device=device,
        pipeline_kwargs=dict(
            max_new_tokens=pipeline_kwargs.get("max_new_tokens", 100),
            temperature=pipeline_kwargs.get("temperature", 0.1),
            top_k=pipeline_kwargs.get("top_k", 35),
        ),
        model_kwargs=model_kwargs
        )

    # Wrap over tokenizer template
    llm = ChatHuggingFace(llm=pipe)

    return llm


def main():
    MODEL_ID = ""
    TASK = "summarization"
    DEVICE = -1 if not torch.cuda.is_available() else 0
    PIPELINE_KWARGS = {
        "max_new_tokens": 100,
        "temperature": 0.1,
        "top_k": 35
    }
    
    # Load LLM
    llm = load_huggingface_llm(
        model_id=MODEL_ID, task=TASK, device=DEVICE, pipeline_kwargs=PIPELINE_KWARGS
    )


if __name__ == "__main__":
    main()