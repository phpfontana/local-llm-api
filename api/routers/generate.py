from fastapi import APIRouter, HTTPException
from api.config import OLLAMA_HOST, OLLAMA_PORT
from api.schemas import GenerateRequest, GenerateResponse
from api.services.llms import load_llm_ollama
from api.services.generate import generate_response


# Instantiate router
router = APIRouter(
    prefix="/api/generate", tags=["generate"], responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=GenerateResponse)
async def main(request: GenerateRequest):
    """
    Generate response.
    """
    try:
        # Load language model
        llm = load_llm_ollama(request.model, base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")
        # Generate response
        response = await generate_response(request.prompt, llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return {"response": response}
