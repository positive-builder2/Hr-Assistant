import os

from hr_assistant import config
from hr_assistant.logger import get_logger
#from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from hr_assistant.embeddings import get_embeddings_model

logger = get_logger(__name__)

def build_vector_store(chunks):
    #logger.info("Embedding %d chunk(s) and building FAISS index.",len(chunks))
    logger.info("Embedding %d chunk(s) and uploading to Qdrant Collection %s...",len(chunks),config.QDRANT_COLLECTION_NAME)
    embeddings_model = get_embeddings_model()
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME
    )
    #logger.info("FAISS index built in Memory")
    logger.info("QDRANT store is built")
    #return FAISS.from_documents(chunks,embeddings_models)
    return vector_store

"""def save_vector_store(vector_store,path:str = config.VECTOR_STORE_PATH):
    vector_store.save_local(path)
    logger.info("Sved FAISS index to '%s'",path)"""

def load_vector_store(path:str = config.VECTOR_STORE_PATH):
    embeddings_model=get_embeddings_model()
    #logger.info("Loading FAISS index from '%s'",path)
    logger.info("Connecting to QDRANT cloud")
    vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings_model,
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            collection_name=config.QDRANT_COLLECTION_NAME
        )
    #return FAISS.load_local(path,embeddings_model,allow_dangerous_deserialization=True)
    return vector_store

def vector_store_exists()->bool:
    #return os.path.exists(os.path.join(path,'index.faiss'))
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    return client.collection_exists(config.QDRANT_COLLECTION_NAME)

def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    logger.info("Creating a retriver with top_k=%d",k)
    return vector_store.as_retriever(search_kwargs={"k": k})