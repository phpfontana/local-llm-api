from fastapi import APIRouter, HTTPException, UploadFile, File
from api.services.document_loaders import *
from api.services.text_splitters import *
from api.services.vectorstore import *
from api.services.embeddings import *
from api.config import CHROMA_PATH
import uuid

# Instantiate router
router = APIRouter(
    prefix="/api/vectorstore", tags=["vectorstore"], responses={404: {"description": "Not found"}},
)

@router.post("/")
async def post(file: UploadFile = File(...)):

    try:
        # Read file contents
        file_name = file.filename
        contents =  await file.read()  
        
        # Split document
        splits = split_text([contents.decode('utf-8')])

        # Load embeddings
        embeddings_model = load_embeddings_model_hf()

        # Load vectorstore
        vectorstore = load_chroma_vectorstore(
            documents=splits, 
            embeddings_model=embeddings_model, 
            persist_directory=CHROMA_PATH, ids=[str(uuid.uuid4()) for _ in splits]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    

    return splits

@router.get("/")
async def get(query_text: str, n_results: int = 5):
    # load embeddings
    embeddings_model = load_embeddings_model_hf()

    # load db
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model)

    # search
    results = db.similarity_search_with_relevance_scores(query=query_text, k=n_results)

    return results

@router.delete("/")
async def delete():
    # load embeddings
    embeddings_model = load_embeddings_model_hf()

    # load db
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model)

    # delete 
    db.delete(ids=[id])

@router.get("/retrieve_all")
async def retrieve_all():
    # load embeddings
    embeddings_model = load_embeddings_model_hf()

    # load db
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model)

    return db._collection.get()

@router.delete("/delete_all")
async def delete_all():
    # load embeddings
    embeddings_model = load_embeddings_model_hf()

    # load db
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model)

    return db.delete_collection()
