from fastapi import APIRouter, HTTPException
from api.config import OLLAMA_URL
from api.schemas.generate import *
from api.services.generate import *


# Instantiate router
router = APIRouter(
    prefix="/api/generate", tags=["generate"], responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=GenerateResponse, status_code=200)
async def main(request: GenerateRequest):
    try:
        llm = load_llm_ollama(request.model, base_url=OLLAMA_URL)
        response = await generate_response(request.prompt, llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return GenerateResponse(response=response)