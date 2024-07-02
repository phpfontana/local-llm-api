from fastapi import APIRouter, HTTPException
from api.config import OLLAMA_URL
from api.schemas.qa import *
from api.services.qa import *

# Instantiate router
router = APIRouter(
    prefix="/api/qa", tags=["qa"], responses={404: {"description": "Not found"}},
)


@router.post("/", response_model= QAResponse, status_code=200)
async def main(request:QARequest):
    try:
        pass    

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return