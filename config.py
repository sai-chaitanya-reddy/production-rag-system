import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # Retrieval
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 5))

    # ChromaDB
    CHROMA_PERSIST_DIR = "./chroma_db"
    COLLECTION_NAME = "rag_documents"

    # Session memory
    MAX_HISTORY = 10

config = Config()
