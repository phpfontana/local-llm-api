from fastapi import FastAPI
from app.routers import rag, embeddings

# Initialize app
app = FastAPI()

# Include routers
app.include_router(router=rag.router)
app.include_router(router=embeddings.router)

@app.get("/")
def root():
    return {"message": "Welcome to the API"}