from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import embeddings, chat

# Initialize app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router=chat.router)
app.include_router(router=embeddings.router)

@app.get("/")
def root():
    return {"message": "Welcome to the API"}