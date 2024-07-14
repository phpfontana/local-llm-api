import dotenv
import os

# Find dotenv path
dotenv_path = dotenv.find_dotenv()

# Load environment variables
dotenv.load_dotenv(dotenv_path)

MODELS_PATH = "models/"