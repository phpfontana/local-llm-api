import dotenv
import os

# Find dotenv path
dotenv_path = dotenv.find_dotenv()

# Load environment variables
dotenv.load_dotenv(dotenv_path)

# Define environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
CHROMA_PATH = "./data/chroma_db"
RAG_TEMPLATE = """
Answer the question if necessary, use the following context to generate the answer:

{context}

---

Answer the question, if necessary, use the following context to generate the answer: {question}
"""