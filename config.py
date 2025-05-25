import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DALL_E_API_KEY = os.getenv("OPENAI_API_KEY")
print(os.getenv("OPENAI_API_KEY"))
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "child_learning")
