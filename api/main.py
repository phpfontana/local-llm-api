from fastapi import FastAPI
from api.routers import rag, embeddings, generate, vectorstore, qa

# Initialize app
app = FastAPI()

# Include routers
app.include_router(router=embeddings.router)
app.include_router(router=generate.router)
app.include_router(router=rag.router)
app.include_router(router=vectorstore.router)
app.include_router(router=qa.router)

@app.get("/")
def root():
    return {"message": "Welcome to the API"}

#TODO: Simplify services