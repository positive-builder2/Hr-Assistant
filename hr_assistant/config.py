import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")


GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

LANGSMITH_TRACING=os.getenv("LANGSMITH_TRACING","false")
LANGSMITH_ENDPOINT=os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT=os.getenv("LANGSMITH_PROJECT")

DATA_FILE_PATH = os.path.join('data','hr_policy.txt')

VECTOR_STORE_PATH = os.path.join('data','faiss_index')


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME")

LLM_MODEL_NAME = 'openai/gpt-oss-20b'


EMBEDDING_MODEL_NAME = 'jina-embeddings-v2-base-en'

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

TOP_K_RESULTS = 3

SYSTEM_PROMPT = (
    "You are a friendly HR assistant. Always use the search_hr_policy tool to look up "
    "facts before answering. If the answer isn't in the search results, say you don't know "
    "instead of guessing."
)


def check_api_keys()->None:
    
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY. Please add it to your .env file.")
    if not JINA_API_KEY:
        raise ValueError("Missing JINA_API_KEY. Please add it to your .env file.")