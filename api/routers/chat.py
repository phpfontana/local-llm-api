import multiprocessing
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.config import MODELS_PATH
from api.schemas.chat import *
from api.services.chat import *

router = APIRouter(
    prefix="/api/chat", tags=["chat"], responses={404: {"description": "Not found"}},
)

model_kwargs = {
    "n_gpu_layers": -1,
    "n_batch": 512,
    "n_ctx": 1024,
    "temperature": 0.8,
    "repeat_penalty": 1.2,
    "top_k": 20,
    "top_p": 0.9,
    "n_threads": multiprocessing.cpu_count() - 1,
    "seed": 42,
    "f16_kv": True,
    "vocab_only": False,
    "use_mlock": False,
    "max_tokens": 1024,
    "verbose": False,
}

@router.post("/", response_model=ChatResponse, status_code=200)
async def main(request: ChatRequest):
    try:
        llm = load_chat_llm(model_name=f"{MODELS_PATH}{request.model}", **model_kwargs)
        chat_messages = format_chat_message(request.messages)
        if not request.stream:
            response = generate_reponse(chat_messages, llm)
            return ChatResponse(
                model=request.model, message=Message(role="assistant", content=response.content), done=True,
            )
        else:
            pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {e}")

""""
{
  "model": "Phi-3-mini-4k-instruct-q4.gguf",
  "messages": [
    {
      "role": "system",
      "content": "You are a chat assistant, always answer briefly and informatively."
    },
   { 
      "role": "user",
      "content": "Hello world"
   }
  ],
  "stream": "false"
}
"""