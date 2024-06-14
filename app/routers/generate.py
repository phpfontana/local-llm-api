from fastapi import APIRouter, HTTPException
from app.schemas import GenerateRequest, GenerateResponse
from app.utils.llms import *

# Instantiate router
router = APIRouter(
    prefix="/api/generate", tags=["generate"], responses={404: {"description": "Not found"}},
)

OLLAMA_BASE_URL = "http://ollama:11434"

@router.post("/", response_model=GenerateResponse)
async def main(request: GenerateRequest):
    """
    Generate response.
    """
    try:
        # Load language model
        llm = load_llm_ollama(request.model, OLLAMA_BASE_URL)
        # Generate response
        response = await generate_response(request.prompt, llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return {"response": response}
