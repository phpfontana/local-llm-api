import dotenv
import os

# Find dotenv path
dotenv_path = dotenv.find_dotenv()

# Load environment variables
dotenv.load_dotenv(dotenv_path)

# Define environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_URI = f"{MILVUS_HOST}:{MILVUS_PORT}"

