import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY") # Added for Groq
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")

    # Updated validation to include GROQ_API_KEY
    if not all([OPENAI_API_KEY, GROQ_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
        raise ValueError("One or more environment variables for OpenAI, Groq, or Pinecone are not set.")

config = Config()