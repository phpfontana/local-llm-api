from fastapi import FastAPI
from api.routers import generate, embeddings, chat, qa

# Initialize app
app = FastAPI()

# Include routers
app.include_router(router=embeddings.router)
app.include_router(router=generate.router)
app.include_router(router=chat.router)
app.include_router(router=qa.router)

@app.get("/")
def root():
    return {"message": "Welcome to the API"}