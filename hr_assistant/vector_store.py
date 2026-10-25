import os

from hr_assistant import config
from hr_assistant.logger import get_logger
from langchain_community.vectorstores import FAISS
from hr_assistant.embeddings import get_embeddings_model

logger = get_logger(__name__)

def build_vector_store(chunks):
    logger.info("Embedding %d chunk(s) and building FAISS index.",len(chunks))
    embeddings_models = get_embeddings_model()
    logger.info("FAISS index built in Memory")
    return FAISS.from_documents(chunks,embeddings_models)

def save_vector_store(vector_store,path:str = config.VECTOR_STORE_PATH):
    vector_store.save_local(path)
    logger.info("Sved FAISS index to '%s'",path)

def load_vector_store(path:str = config.VECTOR_STORE_PATH):
    embeddings_model=get_embeddings_model()
    logger.info("Loading FAISS index from '%s'",path)
    return FAISS.load_local(path,embeddings_model,allow_dangerous_deserialization=True)

def vector_store_exists(path:str = config.VECTOR_STORE_PATH)->bool:
    return os.path.exists(os.path.join(path,'index.faiss'))

def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    logger.info("Creating a retriver with top_k=%d",k)
    return vector_store.as_retriever(search_kwargs={"k": k})