from fastapi import APIRouter, HTTPException
from app.config import OLLAMA_HOST, OLLAMA_PORT
from app.schemas import GenerateRequest, GenerateResponse
from app.services.llms import load_llm_ollama
from app.services.generate import generate_response

# Instantiate router
router = APIRouter(
    prefix="/api/generate", tags=["generate"], responses={404: {"description": "Not found"}},
)

OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

@router.post("/", response_model=GenerateResponse)
async def main(request: GenerateRequest):
    """
    Generate response.
    """
    try:
        # Load language model
        llm = load_llm_ollama(request.model, OLLAMA_URL)
        # Generate response
        response = await generate_response(request.prompt, llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return {"response": response}
